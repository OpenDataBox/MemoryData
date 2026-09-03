"""
Enroot Environment for Harbor.

Enroot is NVIDIA's container runtime designed for HPC/SLURM environments.
It provides Docker-compatible container execution without requiring root privileges.

Key differences from Docker:
- Uses squashfs (.sqsh) images instead of Docker layers
- Stateless containers (no persistent state between starts)
- Designed for SLURM batch systems with GPU support
- No daemon required (rootless)

Usage:
    1. Import Docker image: enroot import docker://image:tag  -> creates image.sqsh
    2. Create container: enroot create --name <name> image.sqsh
    3. Run command: enroot start <name> <command>
"""

import asyncio
import logging
import os
import shlex
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import ClassVar

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths


class EnrootEnvironment(BaseEnvironment):
    """
    Environment that uses NVIDIA Enroot for container execution.
    
    Enroot is commonly available on HPC clusters where Docker is not permitted
    due to security concerns. It provides rootless container execution with
    native GPU support through SLURM.
    """
    
    # Default paths for enroot cache and data
    # Use ENROOT_CACHE_PATH (standard enroot env var) for sqsh images
    _DEFAULT_SQSH_DIR: ClassVar[str] = os.environ.get(
        "ENROOT_CACHE_PATH", 
        os.path.expanduser("~/.cache/enroot")
    )
    _DEFAULT_DATA_DIR: ClassVar[str] = os.environ.get(
        "ENROOT_DATA_PATH",
        os.path.expanduser("~/.local/share/enroot")
    )
    
    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        sqsh_dir: str | None = None,
        data_dir: str | None = None,
        keep_containers: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )
        
        self._sqsh_dir = Path(sqsh_dir or self._DEFAULT_SQSH_DIR)
        self._data_dir = Path(data_dir or self._DEFAULT_DATA_DIR)
        self._keep_containers = keep_containers
        
        # Container state
        self._container_name: str | None = None
        self._run_uuid = uuid.uuid4().hex[:12]
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        
        # Ensure directories exist
        self._sqsh_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        # Mount paths for logging directories
        self._mounts: list[str] = []
        
    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.ENROOT
    
    @property
    def supports_gpus(self) -> bool:
        # Enroot has native GPU support through SLURM
        return True
    
    @property
    def can_disable_internet(self) -> bool:
        # Network isolation is possible but requires special configuration
        return False
    
    @property
    def is_mounted(self) -> bool:
        # We'll mount logging directories
        return True
    
    def _validate_definition(self):
        """Validate that we have either a Dockerfile or docker_image config."""
        dockerfile_path = self.environment_dir / "Dockerfile"
        has_dockerfile = dockerfile_path.exists()
        has_prebuilt = bool(self.task_env_config.docker_image)
        
        if not has_dockerfile and not has_prebuilt:
            raise FileNotFoundError(
                f"No Dockerfile found at {dockerfile_path} and no docker_image "
                "specified in task config. Enroot requires either a Dockerfile "
                "to build from or a prebuilt Docker image reference."
            )
    
    @property
    def _sqsh_path(self) -> Path:
        """Path to the squashfs image file."""
        # Use docker image name or environment name
        image_name = self.task_env_config.docker_image or self.environment_name
        # Sanitize for filesystem
        safe_name = image_name.replace("/", "+").replace(":", "+")
        return self._sqsh_dir / f"{safe_name}.sqsh"
    
    async def _run_command(
        self,
        cmd: list[str],
        timeout_sec: int | None = None,
        env: dict[str, str] | None = None,
        check: bool = True,
    ) -> ExecResult:
        """Run a command and return the result."""
        self.logger.debug(f"Running command: {shlex.join(cmd)}")
        
        merged_env = os.environ.copy()
        merged_env["ENROOT_DATA_PATH"] = str(self._data_dir)
        if env:
            merged_env.update(env)
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=merged_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        try:
            if timeout_sec:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.TimeoutError:
            process.terminate()
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=5
                )
            except asyncio.TimeoutError:
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        
        # Filter noisy enroot-mount messages from stderr
        stderr_str = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
        stderr_filtered = "\n".join(
            line for line in stderr_str.splitlines() 
            if "enroot-mount" not in line
        )
        
        result = ExecResult(
            stdout=stdout_bytes.decode(errors="replace") if stdout_bytes else None,
            stderr=stderr_filtered or None,
            return_code=process.returncode or 0,
        )
        
        if check and result.return_code != 0:
            raise RuntimeError(
                f"Command failed: {shlex.join(cmd)}. "
                f"Return code: {result.return_code}. "
                f"Stdout: {result.stdout}. "
                f"Stderr: {result.stderr}."
            )
        
        return result
    
    async def _import_image(self, docker_image: str) -> Path:
        """Import a Docker image to squashfs format.
        
        Uses Docker/Podman to pull the image first (which handles authentication),
        then exports to tar and imports to enroot.
        """
        if self._sqsh_path.exists():
            self.logger.info(f"Using cached sqsh image: {self._sqsh_path}")
            return self._sqsh_path
        
        self.logger.info(f"Importing Docker image: {docker_image}")
        
        # First try direct enroot import (works for public images on docker.io)
        # Convert docker image reference to enroot URI
        if not docker_image.startswith("docker://"):
            enroot_uri = f"docker://{docker_image}"
        else:
            enroot_uri = docker_image
        
        tmp_sqsh = self._sqsh_path.with_suffix(".sqsh.tmp")
        
        try:
            # Try direct enroot import first (faster for public docker.io images)
            # Note: enroot may return non-zero exit code even on success
            tmp_sqsh.unlink(missing_ok=True)
            await self._run_command(
                ["enroot", "import", "-o", str(tmp_sqsh), enroot_uri],
                timeout_sec=1800,
                check=False,
            )
            if tmp_sqsh.exists():
                tmp_sqsh.rename(self._sqsh_path)
                self.logger.info(f"Successfully imported image to: {self._sqsh_path}")
                return self._sqsh_path
        except Exception as e:
            self.logger.warning(f"Direct enroot import failed: {e}")
        
        tmp_sqsh.unlink(missing_ok=True)
        self.logger.warning("Direct enroot import failed, falling back to docker pull")
        
        # Fallback: Use Docker/Podman to pull (handles auth via docker login)
        try:
            self.logger.info(f"Pulling image via docker: {docker_image}")
            await self._run_command(
                ["docker", "pull", docker_image],
                timeout_sec=1800,
            )
            
            # Import from local podman/docker repository
            self.logger.info(f"Importing from local container registry: {docker_image}")
            # Note: enroot may return non-zero exit code even on success with podman
            await self._run_command(
                ["enroot", "import", "-o", str(tmp_sqsh), f"podman://{docker_image}"],
                timeout_sec=1800,
                check=False,
            )
            
            if not tmp_sqsh.exists():
                # Try dockerd:// as fallback
                await self._run_command(
                    ["enroot", "import", "-o", str(tmp_sqsh), f"dockerd://{docker_image}"],
                    timeout_sec=1800,
                    check=False,
                )
            
            if not tmp_sqsh.exists():
                raise RuntimeError("Both podman:// and dockerd:// import failed")
            
            tmp_sqsh.rename(self._sqsh_path)
            self.logger.info(f"Successfully imported image to: {self._sqsh_path}")
        except Exception as e:
            tmp_sqsh.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to import Docker image: {e}") from e
        
        return self._sqsh_path
    
    async def _build_image(self) -> Path:
        """Build a Docker image from Dockerfile and convert to sqsh."""
        dockerfile_path = self.environment_dir / "Dockerfile"
        
        if self._sqsh_path.exists():
            self.logger.info(f"Using cached sqsh image: {self._sqsh_path}")
            return self._sqsh_path
        
        self.logger.info(f"Building image from Dockerfile: {dockerfile_path}")
        
        # Build with Docker/Podman first
        image_tag = f"enroot-build-{self._run_uuid}"
        
        await self._run_command(
            ["docker", "build", "-t", image_tag, str(self.environment_dir)],
            timeout_sec=1800,
        )
        
        # Import from podman/docker daemon to enroot
        # podman:// imports from local podman repository
        # dockerd:// imports from local docker daemon
        tmp_sqsh = self._sqsh_path.with_suffix(".sqsh.tmp")
        
        try:
            # Try podman:// first (our docker is aliased to podman)
            self.logger.info(f"Importing from local container registry: {image_tag}")
            # Clean up any leftover tmp file first
            tmp_sqsh.unlink(missing_ok=True)
            
            # Note: enroot may return non-zero exit code even on success with podman
            # So we check if the file exists rather than relying on exit code
            result = await self._run_command(
                ["enroot", "import", "-o", str(tmp_sqsh), f"podman://{image_tag}"],
                timeout_sec=1800,
                check=False,  # Don't raise on non-zero exit
            )
            
            if not tmp_sqsh.exists():
                # If file doesn't exist, the import truly failed
                self.logger.info(f"podman:// failed, trying dockerd:// scheme...")
                result = await self._run_command(
                    ["enroot", "import", "-o", str(tmp_sqsh), f"dockerd://{image_tag}"],
                    timeout_sec=1800,
                    check=False,
                )
            
            if not tmp_sqsh.exists():
                raise RuntimeError(
                    f"Failed to import image. podman:// and dockerd:// both failed. "
                    f"Last stderr: {result.stderr}"
                )
            
            tmp_sqsh.rename(self._sqsh_path)
            self.logger.info(f"Successfully built and imported image to: {self._sqsh_path}")
        except Exception as e:
            tmp_sqsh.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to build image: {e}") from e
        finally:
            # Clean up the docker image
            try:
                await self._run_command(
                    ["docker", "rmi", "-f", image_tag],
                    check=False,
                )
            except Exception:
                pass
        
        return self._sqsh_path
    
    async def _create_container(self) -> str:
        """Create a container from the sqsh image."""
        container_name = f"hb-{self.environment_name}-{self._run_uuid}"
        
        self.logger.info(f"Creating container: {container_name}")
        
        await self._run_command(
            ["enroot", "create", "--name", container_name, str(self._sqsh_path)],
            timeout_sec=300,
        )
        
        return container_name
    
    async def start(self, force_build: bool) -> None:
        """Start the enroot environment."""
        # Determine if we should use prebuilt image
        use_prebuilt = not force_build and self.task_env_config.docker_image
        
        if use_prebuilt:
            await self._import_image(self.task_env_config.docker_image)
        else:
            await self._build_image()
        
        # Create container
        self._container_name = await self._create_container()
        
        # Create a persistent tmpdir for tmux sockets to survive across exec calls
        # Enroot is stateless - each 'enroot start' is a new process with fresh /tmp
        # By mounting a host tmpdir to /tmp, tmux sessions persist between exec calls
        self._tmpdir = tempfile.TemporaryDirectory(prefix="enroot-tmp-")
        
        # Set up mounts for logging directories
        self._mounts = [
            f"{self.trial_paths.verifier_dir}:{EnvironmentPaths.verifier_dir}",
            f"{self.trial_paths.agent_dir}:{EnvironmentPaths.agent_dir}",
            f"{self._tmpdir.name}:/tmp",  # Persist tmux sockets across exec calls
        ]
        
        self.logger.info(f"Container {self._container_name} ready")
    
    async def stop(self, delete: bool) -> None:
        """Stop and optionally delete the container."""
        # Clean up the tmpdir
        if self._tmpdir:
            try:
                self._tmpdir.cleanup()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup tmpdir: {e}")
            self._tmpdir = None
        
        if not self._container_name:
            return
        
        if self._keep_containers and delete:
            self.logger.warning(
                "Both `keep_containers` and `delete` are set. "
                "keep_containers takes precedence."
            )
        
        if not self._keep_containers:
            self.logger.info(f"Removing container: {self._container_name}")
            try:
                await self._run_command(
                    ["enroot", "remove", "-f", self._container_name],
                    check=False,
                )
            except Exception as e:
                self.logger.warning(f"Failed to remove container: {e}")
        
        self._container_name = None
    
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        """Execute a command in the container."""
        if not self._container_name:
            raise RuntimeError("Container not started")
        
        # Build the command
        exec_cmd = ["enroot", "start", "--rw"]
        
        # Add mounts
        for mount in self._mounts:
            exec_cmd.extend(["--mount", mount])
        
        exec_cmd.append(self._container_name)
        
        # Build bash command with optional cwd
        if cwd:
            bash_cmd = f"cd {shlex.quote(cwd)} && {command}"
        else:
            bash_cmd = command
        
        exec_cmd.extend(["bash", "-c", bash_cmd])
        
        return await self._run_command(
            exec_cmd,
            timeout_sec=timeout_sec,
            env=env,
            check=False,
        )
    
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Upload a file to the container."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        # Enroot containers are stateless, so we need to use a mount
        # or copy during exec. We'll use a temp mount approach.
        # For now, copy via exec with base64 encoding
        import base64
        
        content = source.read_bytes()
        encoded = base64.b64encode(content).decode()
        
        # Create parent directory and decode file
        await self.exec(
            f"mkdir -p $(dirname {shlex.quote(target_path)}) && "
            f"echo '{encoded}' | base64 -d > {shlex.quote(target_path)}",
            timeout_sec=60,
        )
    
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Upload a directory to the container."""
        source = Path(source_dir)
        if not source.is_dir():
            raise NotADirectoryError(f"Source is not a directory: {source}")
        
        # Create a tarball and extract in container
        import tarfile
        import io
        import base64
        
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            tar.add(source, arcname=".")
        
        encoded = base64.b64encode(buffer.getvalue()).decode()
        
        await self.exec(
            f"mkdir -p {shlex.quote(target_dir)} && "
            f"cd {shlex.quote(target_dir)} && "
            f"echo '{encoded}' | base64 -d | tar xzf -",
            timeout_sec=120,
        )
    
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Download a file from the container."""
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        
        import base64
        
        # Read and encode in container
        result = await self.exec(
            f"base64 {shlex.quote(source_path)}",
            timeout_sec=60,
        )
        
        if result.return_code != 0:
            raise RuntimeError(f"Failed to read file: {result.stderr}")
        
        content = base64.b64decode(result.stdout.strip())
        target.write_bytes(content)
    
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download a directory from the container."""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        
        import base64
        import tarfile
        import io
        
        # Create tarball in container and download
        result = await self.exec(
            f"cd {shlex.quote(source_dir)} && tar czf - . | base64",
            timeout_sec=120,
        )
        
        if result.return_code != 0:
            raise RuntimeError(f"Failed to read directory: {result.stderr}")
        
        content = base64.b64decode(result.stdout.strip())
        buffer = io.BytesIO(content)
        
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            tar.extractall(target)
    
    async def attach(self) -> None:
        """Attach to the container interactively."""
        if not self._container_name:
            raise RuntimeError("Container not started")
        
        mount_args = []
        for mount in self._mounts:
            mount_args.extend(["--mount", mount])
        
        os.execvp(
            "enroot",
            ["enroot", "start", "--rw", *mount_args, self._container_name, "bash"],
        )
