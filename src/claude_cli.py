import os
import tempfile
import atexit
import shutil
import asyncio
import json
from typing import AsyncGenerator, Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ClaudeCodeCLI:
    def __init__(self, timeout: int = 600000, cwd: Optional[str] = None, cli_path: Optional[str] = None):
        self.timeout = timeout / 1000  # Convert ms to seconds
        self.temp_dir = None
        self.cli_path = cli_path or "claude"

        # If cwd is provided (from CLAUDE_CWD env var), use it
        # Otherwise create an isolated temp directory
        if cwd:
            self.cwd = Path(cwd)
            # Check if the directory exists
            if not self.cwd.exists():
                logger.error(f"ERROR: Specified working directory does not exist: {self.cwd}")
                raise ValueError(f"Working directory does not exist: {self.cwd}")
            else:
                logger.info(f"Using CLAUDE_CWD: {self.cwd}")
        else:
            # Create isolated temp directory (cross-platform)
            self.temp_dir = tempfile.mkdtemp(prefix="claude_code_workspace_")
            self.cwd = Path(self.temp_dir)
            logger.info(f"Using temporary isolated workspace: {self.cwd}")
            atexit.register(self._cleanup_temp_dir)

        # Import auth manager
        from src.auth import auth_manager, validate_claude_code_auth

        # Validate authentication
        is_valid, _ = validate_claude_code_auth()
        if not is_valid:
            logger.warning("Claude Code authentication issues detected")

        # Store auth environment variables
        self.claude_env_vars = auth_manager.get_claude_code_env_vars()

    async def _execute_cli(
        self, 
        prompt: str, 
        model: Optional[str] = None, 
        system_prompt: Optional[str] = None,
        permission_mode: Optional[str] = None,
        stream: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:


        """Directly execute Claude CLI and yield JSON events from stdout."""
        
        # Build command
        cmd = [self.cli_path, "--output-format", "stream-json", "--verbose", "--print"]
        
        # 🆕 Key flag for granular streaming (token-by-token)
        if stream:
            cmd.append("--include-partial-messages")

        
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
            
        if permission_mode:
            cmd.extend(["--permission-mode", permission_mode])
            
        cmd.extend(["--", prompt])

        
        # Build environment
        env = {**os.environ, **self.claude_env_vars}
        if model:
            env["ANTHROPIC_MODEL"] = model
            
        logger.debug(f"Executing CLI: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(self.cwd)
        )

        # Task to log stderr
        async def log_stderr(stderr):
            while True:
                line = await stderr.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                if line_str:
                    logger.error(f"  [CLI STDERR] {line_str}")

        stderr_task = asyncio.create_task(log_stderr(process.stderr))

        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                line_str = line.decode().strip()
                if not line_str:
                    continue
                
                try:
                    # Parse JSON event
                    event = json.loads(line_str)
                    yield event
                except json.JSONDecodeError:
                    # Ignore non-JSON lines (warnings, etc.)
                    if not line_str.startswith("{"):
                        logger.debug(f"Non-JSON CLI output: {line_str}")
                        continue
                    logger.warning(f"Failed to parse CLI JSON: {line_str}")

            await process.wait()
            if process.returncode != 0:
                logger.error(f"CLI process exited with code {process.returncode}")

        finally:
            # Cleanup
            if process.returncode is None:
                try:
                    process.terminate()
                except:
                    pass
            await stderr_task

    async def verify_cli(self) -> bool:
        """Verify Claude CLI is working and authenticated."""
        try:
            logger.info("Verifying Claude CLI connection...")
            
            async for event in self._execute_cli(prompt="ready"):
                # If we get any valid success results, it's working
                if event.get("type") == "result" and not event.get("is_error", False):
                    logger.info("✅ Claude CLI verified successfully")
                    return True
                
                # If we see assistant response starting, that's also good
                if event.get("type") == "assistant":
                    logger.info("✅ Claude CLI verified successfully (assistant response detected)")
                    return True

            logger.warning("⚠️ Claude CLI verification returned no success events")
            return False

        except Exception as e:
            logger.error(f"Claude CLI verification failed: {e}")
            return False

    async def run_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = True,
        max_turns: int = 10,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        permission_mode: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run completion using direct CLI integration."""
        
        # Note: We currently focus on the main conversation flow. 
        # Advanced SDK features like session_id/continue_session would need
        # additional CLI flags or session management logic if required.
        
        try:
            async for event in self._execute_cli(
                prompt=prompt, 
                model=model, 
                system_prompt=system_prompt,
                permission_mode=permission_mode,
                stream=stream
            ):
                yield event


        except Exception as e:
            logger.error(f"Error during CLI completion: {e}")
            yield {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "error_message": str(e),
            }

    def parse_claude_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the assistant message from CLI events."""
        # Check for result event
        for event in messages:
            if event.get("type") == "result" and "result" in event:
                return event["result"]

        # Collect text from assistant events
        last_text = None
        for event in messages:
            if event.get("type") == "assistant" and "message" in event:
                msg = event["message"]
                if "content" in msg:
                    content = msg["content"]
                    if isinstance(content, list):
                        parts = []
                        for block in content:
                            if block.get("type") == "text":
                                parts.append(block.get("text", ""))
                        if parts:
                            last_text = "".join(parts)
                    elif isinstance(content, str):
                        last_text = content
        return last_text

    def extract_metadata(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata from CLI events."""
        metadata = {
            "session_id": None,
            "total_cost_usd": 0.0,
            "duration_ms": 0,
            "num_turns": 0,
            "model": None,
        }

        for event in messages:
            if event.get("type") == "result":
                metadata.update({
                    "total_cost_usd": event.get("total_cost_usd", 0.0),
                    "duration_ms": event.get("duration_ms", 0),
                    "num_turns": event.get("num_turns", 0),
                    "session_id": event.get("session_id"),
                })
            elif event.get("type") == "system" and event.get("subtype") == "init":
                metadata.update({
                    "session_id": event.get("session_id"),
                    "model": event.get("model")
                })

        return metadata

    def estimate_token_usage(
        self, prompt: str, completion: str, model: Optional[str] = None
    ) -> Dict[str, int]:
        """Estimate token usage."""
        p_tokens = max(1, len(prompt) // 4)
        c_tokens = max(1, len(completion) // 4)
        return {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
        }

    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass
