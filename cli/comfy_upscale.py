import os
import json
import argparse
import requests
import time
import websocket
import uuid
import sys
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import io
from tqdm import tqdm
from pyfiglet import figlet_format
from halo import Halo


class Logger:
    """Handles logging with verbose mode support."""

    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def info(self, message: str):
        """Always print info messages."""
        print(f"ℹ️  {message}")

    def success(self, message: str):
        """Print success messages."""
        print(f"✅ {message}")

    def warning(self, message: str):
        """Print warning messages."""
        print(f"⚠️  {message}")

    def error(self, message: str):
        """Print error messages."""
        print(f"❌ {message}")

    def debug(self, message: str):
        """Print debug messages only in verbose mode."""
        if self._verbose:
            print(f"🔍 [DEBUG] {message}")

    def verbose(self, message: str, data: Optional[Dict] = None):
        """Print verbose messages with optional JSON data."""
        if self._verbose:
            print(f"📝 [VERBOSE] {message}")
            if data:
                print(f"   └─ {json.dumps(data, indent=6)}")


class WorkflowConverter:
    """Handles conversion of workflow format to API format."""

    def __init__(self, logger: Logger):
        self.logger = logger

    def create_link_map(self, workflow_data: Dict) -> Dict:
        """Create a mapping of link IDs to source node and output index."""
        link_map = {}
        links = workflow_data.get("links", [])
        self.logger.debug(f"Processing {len(links)} workflow links")

        for link in links:
            link_id, source_node_id, source_output_index = (
                link[0],
                str(link[1]),
                link[2],
            )
            link_map[link_id] = (source_node_id, source_output_index)
            self.logger.verbose(
                f"Link mapped: {link_id} -> Node {source_node_id}, Output {source_output_index}"
            )

        return link_map

    def process_node_inputs(self, node: Dict, link_map: Dict) -> Dict:
        """Process input links for a node and return input dictionary."""
        inputs = {}
        for input_info in node.get("inputs", []):
            if isinstance(input_info, dict):
                input_name = input_info.get("name")
                link_id = input_info.get("link")
                if input_name and link_id in link_map:
                    source_node_id, source_output_index = link_map[link_id]
                    inputs[input_name] = [source_node_id, source_output_index]
                    self.logger.verbose(
                        f"Input '{input_name}' connected to Node {source_node_id}"
                    )
        return inputs

    def handle_node_widgets(
        self, node: Dict, uploaded_filename: Optional[str] = None
    ) -> Dict:
        """Handle widget values based on node type."""
        node_type = node["type"]
        inputs = {}

        if "widgets_values" not in node:
            return inputs

        self.logger.debug(f"Processing widgets for node type: {node_type}")
        widgets = node["widgets_values"]

        if node_type == "LoadImage":
            inputs["image"] = uploaded_filename if uploaded_filename else widgets[0]
            inputs["upload"] = "image"
            self.logger.verbose(f"LoadImage configured with: {inputs['image']}")

        elif node_type == "SaveImage":
            inputs["filename_prefix"] = widgets[0]
            self.logger.verbose(f"SaveImage prefix: {widgets[0]}")

        elif node_type == "CLIPTextEncode":
            inputs["text"] = widgets[0]
            self.logger.verbose(f"CLIPTextEncode prompt: {widgets[0][:50]}...")

        elif node_type == "CheckpointLoaderSimple":
            inputs["ckpt_name"] = widgets[0]
            self.logger.verbose(f"Checkpoint loaded: {widgets[0]}")

        elif node_type == "UpscaleModelLoader":
            inputs["model_name"] = widgets[0]
            self.logger.verbose(f"Upscale model: {widgets[0]}")

        elif node_type == "UltimateSDUpscale":
            param_names = [
                "upscale_by",
                "seed",
                "control_after_generate",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "denoise",
                "mode_type",
                "tile_width",
                "tile_height",
                "mask_blur",
                "tile_padding",
                "seam_fix_mode",
                "seam_fix_denoise",
                "seam_fix_width",
                "seam_fix_mask_blur",
                "seam_fix_padding",
                "force_uniform_tiles",
                "tiled_decode",
            ]

            for i, val in enumerate(widgets):
                if i < len(param_names):
                    inputs[param_names[i]] = val
                else:
                    inputs[f"_{i}"] = val

            # Apply fixes and defaults
            inputs["mask_blur"] = inputs.get("seam_fix_mask_blur", 4)
            if "seam_fix_mask_blur" not in inputs:
                inputs["seam_fix_mask_blur"] = 4
            if "tile_height" in inputs and inputs["tile_height"] < 64:
                self.logger.debug(
                    f"Adjusting tile_height from {inputs['tile_height']} to 64"
                )
                inputs["tile_height"] = 64

            self.logger.verbose(
                "UltimateSDUpscale parameters",
                {
                    "upscale_by": inputs.get("upscale_by"),
                    "steps": inputs.get("steps"),
                    "denoise": inputs.get("denoise"),
                    "tile_size": f"{inputs.get('tile_width')}x{inputs.get('tile_height')}",
                },
            )

        return inputs

    def convert(
        self, workflow_data: Dict, uploaded_filename: Optional[str] = None
    ) -> Dict:
        """Convert workflow data to API format."""
        self.logger.debug("Converting workflow to API format")
        api_format = {}
        link_map = self.create_link_map(workflow_data)

        for node in workflow_data.get("nodes", []):
            node_id = str(node["id"])
            node_type = node["type"]

            node_inputs = self.process_node_inputs(node, link_map)
            widget_inputs = self.handle_node_widgets(node, uploaded_filename)
            node_inputs.update(widget_inputs)

            api_format[node_id] = {"class_type": node_type, "inputs": node_inputs}
            self.logger.verbose(f"Node {node_id} ({node_type}) processed")

        self.logger.debug(f"Workflow conversion complete: {len(api_format)} nodes")
        return api_format


class ComfyUIClient:
    """Client for interacting with ComfyUI server."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = 8188, logger: Optional[Logger] = None
    ):
        self.host = host
        self.port = port
        self.logger = logger or Logger()
        self.client_id = str(uuid.uuid4())
        self.ws = None

        self.logger.debug(f"Initializing ComfyUI client with ID: {self.client_id}")
        self._connect()

    def _check_server(self) -> bool:
        """Check if ComfyUI server is accessible."""
        try:
            url = f"http://{self.host}:{self.port}/system_stats"
            self.logger.debug(f"Checking server at: {url}")
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"Server check failed: {str(e)}")
            return False

    def _connect(self):
        """Connect to ComfyUI server and WebSocket."""
        spinner = Halo(text="Connecting to ComfyUI server", spinner="dots")
        spinner.start()

        if not self._check_server():
            spinner.fail("Cannot connect to ComfyUI server")
            self.logger.error(
                f"Server not accessible at http://{self.host}:{self.port}"
            )
            self.logger.info("Please ensure ComfyUI is running")
            sys.exit(1)

        try:
            self.ws = websocket.WebSocket()
            ws_url = f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}"
            self.logger.debug(f"Connecting to WebSocket: {ws_url}")
            self.ws.connect(ws_url)
            spinner.succeed(f"Connected to ComfyUI at {self.host}:{self.port}")
        except Exception as e:
            spinner.fail("WebSocket connection failed")
            self.logger.error(f"WebSocket error: {str(e)}")
            sys.exit(1)

    def upload_image(self, image_path: str) -> str:
        """Upload an image to ComfyUI server."""
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found: {image_path}")
            sys.exit(1)

        filename = os.path.basename(image_path)
        spinner = Halo(text=f"Uploading {filename}", spinner="dots")
        spinner.start()

        try:
            with open(image_path, "rb") as f:
                files = {"image": (filename, f, "image/png")}
                url = f"http://{self.host}:{self.port}/upload/image"
                self.logger.debug(f"Uploading to: {url}")
                response = requests.post(url, files=files)
                response.raise_for_status()
                spinner.succeed(f"Uploaded {filename}")
                self.logger.verbose("Upload response", response.json())
                return filename
        except requests.exceptions.RequestException as e:
            spinner.fail("Upload failed")
            self.logger.error(f"Upload error: {str(e)}")
            sys.exit(1)

    def queue_prompt(self, workflow: Dict) -> Dict:
        """Queue a workflow prompt for execution."""
        try:
            prompt_data = {"prompt": workflow, "client_id": self.client_id}
            url = f"http://{self.host}:{self.port}/prompt"

            self.logger.debug("Queueing prompt to server")
            self.logger.verbose(
                "Prompt data", {"client_id": self.client_id, "nodes": len(workflow)}
            )

            response = requests.post(url, json=prompt_data)
            response.raise_for_status()
            result = response.json()

            self.logger.verbose("Server response", result)
            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to queue prompt: {str(e)}")
            if "response" in locals():
                self.logger.debug(f"Response content: {response.text}")
            sys.exit(1)

    def get_image(
        self, filename: str, subfolder: str = "", folder_type: str = "output"
    ) -> Optional[requests.Response]:
        """Retrieve an image from ComfyUI server."""
        try:
            url = f"http://{self.host}:{self.port}/view?filename={filename}&subfolder={subfolder}&type={folder_type}"
            self.logger.debug(f"Fetching image: {url}")
            response = requests.get(url)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"Failed to get image {filename}: {str(e)}")
            return None

    def wait_for_completion(self):
        """Wait for workflow execution to complete with progress tracking."""
        with tqdm(
            total=100,
            desc="Processing",
            unit="%",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            leave=False,
        ) as pbar:
            while True:
                try:
                    out = self.ws.recv()
                    if isinstance(out, str):
                        message = json.loads(out)

                        self.logger.verbose("WebSocket message received", message)

                        if message.get("type") == "executing":
                            data = message.get("data", {})
                            node = data.get("node")

                            if node is None:
                                pbar.n = 100
                                pbar.set_description("✓ Complete")
                                pbar.refresh()
                                break
                            else:
                                self.logger.debug(f"Executing node: {node}")
                                if pbar.n < 95:
                                    pbar.update(5)

                        elif message.get("type") == "progress":
                            value = message.get("data", {}).get("value", 0)
                            max_val = message.get("data", {}).get("max", 100)
                            progress = (
                                int((value / max_val) * 100) if max_val > 0 else 0
                            )
                            pbar.n = min(progress, 95)
                            pbar.refresh()

                    time.sleep(0.1)
                except websocket.WebSocketConnectionClosedException:
                    self.logger.error("WebSocket connection closed unexpectedly")
                    sys.exit(1)

    def get_history(self, prompt_id: str) -> Optional[Dict]:
        """Get execution history for a prompt."""
        try:
            url = f"http://{self.host}:{self.port}/history"
            self.logger.debug(f"Fetching history for prompt: {prompt_id}")
            response = requests.get(url)
            response.raise_for_status()
            history = response.json()

            if prompt_id in history:
                self.logger.verbose("History retrieved", {"prompt_id": prompt_id})
                return history[prompt_id]
            else:
                self.logger.warning(f"Prompt {prompt_id} not found in history")
                return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get history: {str(e)}")
            return None

    def search_recent_images(
        self, pattern_prefix: str = "ComfyUI"
    ) -> Optional[requests.Response]:
        """Search for recent images using pattern matching as fallback."""
        self.logger.debug("Searching for recent images using pattern matching")

        for i in range(100, 0, -1):
            for pattern in [
                f"{pattern_prefix}_{i:05d}_.png",
                f"{pattern_prefix}_{i:04d}_.png",
            ]:
                output_image = self.get_image(pattern)
                if output_image and output_image.status_code == 200:
                    self.logger.debug(f"Found image using pattern: {pattern}")
                    return output_image

        self.logger.warning("No images found using pattern matching")
        return None

    def close(self):
        """Close WebSocket connection."""
        if self.ws:
            self.logger.debug("Closing WebSocket connection")
            self.ws.close()


class ImageProcessor:
    """Handles image processing workflow."""

    def __init__(
        self, client: ComfyUIClient, converter: WorkflowConverter, logger: Logger
    ):
        self.client = client
        self.converter = converter
        self.logger = logger

    def process(
        self,
        workflow_data: Dict,
        input_path: str,
        output_dir: str,
        prefix: str = "",
        output_format: Optional[str] = None,
    ) -> str:
        """Process a single image through the workflow."""
        try:
            # Upload image
            uploaded_filename = self.client.upload_image(input_path)

            # Convert workflow
            workflow_api = self.converter.convert(workflow_data, uploaded_filename)

            # Queue and execute
            self.logger.info(f"Processing: {os.path.basename(input_path)}")
            prompt_response = self.client.queue_prompt(workflow_api)
            prompt_id = prompt_response["prompt_id"]
            self.logger.debug(f"Prompt ID: {prompt_id}")

            # Wait for completion
            self.client.wait_for_completion()
            time.sleep(2)  # Brief pause for file system

            # Get preview node
            preview_node_id = self._find_preview_node(workflow_api)

            # Retrieve output with fallback
            output_image = self._retrieve_output_with_fallback(
                prompt_id, preview_node_id
            )
            if not output_image:
                self.logger.error("Could not retrieve output image")
                raise Exception("Output retrieval failed")

            # Save output
            output_path = self._save_output(
                output_image, input_path, output_dir, prefix, output_format
            )

            # Cleanup
            self._cleanup_input_folder()

            return output_path

        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            raise

    def _find_preview_node(self, workflow_api: Dict) -> str:
        """Find the PreviewImage node ID in the workflow."""
        for node_id, node in workflow_api.items():
            if node["class_type"] == "PreviewImage":
                self.logger.debug(f"Preview node found: {node_id}")
                return node_id

        self.logger.error("No PreviewImage node found in workflow")
        sys.exit(1)

    def _retrieve_output_with_fallback(
        self, prompt_id: str, preview_node_id: str
    ) -> Optional[requests.Response]:
        """Retrieve output image with fallback to pattern search."""
        spinner = Halo(text="Retrieving output", spinner="dots")
        spinner.start()

        # Try history-based retrieval first
        prompt_data = self.client.get_history(prompt_id)
        if prompt_data:
            outputs = prompt_data.get("outputs", {})
            node_output = outputs.get(preview_node_id, {})
            images = node_output.get("images", [])

            if images:
                img_info = images[-1]
                if isinstance(img_info, dict):
                    filename = img_info["filename"]
                    subfolder = img_info.get("subfolder", "")
                    folder_type = img_info.get("type", "output")
                else:
                    filename = img_info
                    subfolder = ""
                    folder_type = "output"

                self.logger.debug(
                    f"Output image: {filename} (subfolder: {subfolder}, type: {folder_type})"
                )

                output_image = self.client.get_image(filename, subfolder, folder_type)
                if output_image and output_image.status_code == 200:
                    spinner.succeed(f"Retrieved output: {filename}")
                    return output_image

        # Fallback to pattern search
        spinner.text = "Searching for output (fallback)"
        self.logger.warning("History retrieval failed, trying pattern search")
        output_image = self.client.search_recent_images()

        if output_image:
            spinner.succeed("Retrieved output using pattern search")
            return output_image

        spinner.fail("Failed to retrieve output")
        return None

    def _save_output(
        self,
        output_image: requests.Response,
        input_path: str,
        output_dir: str,
        prefix: str,
        output_format: Optional[str],
    ) -> str:
        """Save the output image with specified format."""
        base_filename = Path(input_path).stem

        # Determine output format
        if output_format:
            ext = output_format.lower()
            if ext == "jpg":
                ext = "jpeg"
        else:
            input_ext = Path(input_path).suffix.lower().lstrip(".")
            if input_ext in ("jpg", "jpeg", "png", "webp"):
                ext = "jpeg" if input_ext in ("jpg", "jpeg") else input_ext
            else:
                ext = "png"

        display_ext = "jpg" if ext == "jpeg" else ext
        output_filename = f"{prefix}{base_filename}.{display_ext}"
        output_path = os.path.join(output_dir, output_filename)

        spinner = Halo(text=f"Saving {output_filename}", spinner="dots")
        spinner.start()

        try:
            img = Image.open(io.BytesIO(output_image.content))

            # Save with appropriate format settings
            if ext == "jpeg":
                img = img.convert("RGB")  # JPEG doesn't support transparency
                img.save(
                    output_path,
                    format="JPEG",
                    quality=100,
                    subsampling="4:4:4",
                    progressive=False,
                )
                self.logger.debug("Saved as JPEG with quality=100")
            elif ext == "png":
                img.save(output_path, format="PNG")
                self.logger.debug("Saved as PNG")
            elif ext == "webp":
                img.save(output_path, format="WEBP", quality=100, method=6)
                self.logger.debug("Saved as WebP with quality=100")
            else:
                img.save(output_path, format="PNG")
                self.logger.debug("Saved as PNG (fallback)")

            spinner.succeed(f"Saved: {output_filename}")
            self.logger.verbose(
                "Output saved",
                {
                    "path": output_path,
                    "format": ext,
                    "size": f"{img.width}x{img.height}",
                },
            )
            return output_path

        except Exception as e:
            spinner.fail("Failed to save output")
            self.logger.error(f"Save error: {str(e)}")
            raise

    def _cleanup_input_folder(self):
        """Clean up uploaded images from input folder."""
        # Use the script's directory to find the ComfyUI input folder
        script_dir = Path(__file__).parent.parent
        input_folder = script_dir / "input"

        if not input_folder.exists():
            self.logger.debug(f"Input folder not found: {input_folder}")
            return

        supported_formats = {".png", ".jpg", ".jpeg", ".webp"}
        deleted_count = 0

        for file_path in input_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    self.logger.debug(f"Deleted: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Could not delete {file_path}: {e}")

        if deleted_count > 0:
            self.logger.debug(f"Cleaned up {deleted_count} file(s) from input folder")


def get_default_workflow() -> Dict:
    """Return the default Ultimate SD Upscale workflow."""
    return {
        "nodes": [
            {
                "id": 1,
                "type": "CheckpointLoaderSimple",
                "outputs": [
                    {"name": "MODEL", "type": "MODEL", "links": [1]},
                    {"name": "CLIP", "type": "CLIP", "links": [6, 7]},
                    {"name": "VAE", "type": "VAE", "links": [2]},
                ],
                "widgets_values": ["realisticmix_iiV12Version12.safetensors"],
            },
            {
                "id": 2,
                "type": "CLIPTextEncode",
                "inputs": [{"name": "clip", "type": "CLIP", "link": 6}],
                "outputs": [
                    {"name": "CONDITIONING", "type": "CONDITIONING", "links": [8]}
                ],
                "widgets_values": ["photo of a person"],
            },
            {
                "id": 3,
                "type": "CLIPTextEncode",
                "inputs": [{"name": "clip", "type": "CLIP", "link": 7}],
                "outputs": [
                    {"name": "CONDITIONING", "type": "CONDITIONING", "links": [9]}
                ],
                "widgets_values": ["blur, bad quality, text"],
            },
            {
                "id": 4,
                "type": "UpscaleModelLoader",
                "outputs": [
                    {"name": "UPSCALE_MODEL", "type": "UPSCALE_MODEL", "links": [3]}
                ],
                "widgets_values": ["RealESRGAN_x2plus.pth"],
            },
            {
                "id": 5,
                "type": "LoadImage",
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [4]}],
                "widgets_values": ["placeholder.png", "image"],
            },
            {
                "id": 6,
                "type": "UltimateSDUpscale",
                "inputs": [
                    {"name": "image", "type": "IMAGE", "link": 4},
                    {"name": "model", "type": "MODEL", "link": 1},
                    {"name": "positive", "type": "CONDITIONING", "link": 8},
                    {"name": "negative", "type": "CONDITIONING", "link": 9},
                    {"name": "vae", "type": "VAE", "link": 2},
                    {"name": "upscale_model", "type": "UPSCALE_MODEL", "link": 3},
                ],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [5]}],
                "widgets_values": [
                    2.00,
                    8267030,
                    "randomize",
                    4,
                    5.0,
                    "euler",
                    "normal",
                    0.20,
                    "Linear",
                    512,
                    512,
                    8,
                    32,
                    "None",
                    1.00,
                    64,
                    8,
                    16,
                    "false",
                    "false",
                ],
            },
            {
                "id": 7,
                "type": "PreviewImage",
                "inputs": [{"name": "images", "type": "IMAGE", "link": 5}],
                "outputs": [],
                "properties": {"Node name for S&R": "PreviewImage"},
            },
        ],
        "links": [
            [1, 1, 0, 6, 1, "MODEL"],
            [2, 1, 2, 6, 4, "VAE"],
            [3, 4, 0, 6, 5, "UPSCALE_MODEL"],
            [4, 5, 0, 6, 0, "IMAGE"],
            [5, 6, 0, 7, 0, "IMAGE"],
            [6, 1, 1, 2, 0, "CLIP"],
            [7, 1, 1, 3, 0, "CLIP"],
            [8, 2, 0, 6, 2, "CONDITIONING"],
            [9, 3, 0, 6, 3, "CONDITIONING"],
        ],
        "version": 0.4,
    }


def print_banner():
    """Print application banner."""
    banner = figlet_format("Upscaler", font="slant")
    print(banner)
    print("=" * 60)
    print("  ComfyUI Ultimate SD Upscale Tool")
    print("  High-Quality Image Enhancement")
    print("=" * 60)
    print()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upscale images using ComfyUI and Ultimate SD Upscale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i image.png                          # Upscale single image
  %(prog)s -i ./images -o ./upscaled            # Upscale directory
  %(prog)s -i image.png -f jpg                   # Output as JPEG
  %(prog)s -i ./images --prefix "upscaled_"     # Add prefix
  %(prog)s -i image.png -v                       # Verbose mode
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        default=os.getcwd(),
        help="Input directory or image file (default: current directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (default: <input>/image_upscale)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="ComfyUI server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8188, help="ComfyUI server port (default: 8188)"
    )
    parser.add_argument(
        "--prefix", default="", help="Prefix for output filenames (default: none)"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["jpg", "png", "webp"],
        help="Output image format (default: same as input)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output with debug information",
    )

    return parser.parse_args()


def get_image_files(input_path: str, logger: Logger) -> List[str]:
    """Get list of image files to process."""
    supported_formats = {".png", ".jpg", ".jpeg", ".webp"}

    if os.path.isfile(input_path):
        if Path(input_path).suffix.lower() not in supported_formats:
            logger.error(f"Unsupported file format: {input_path}")
            sys.exit(1)
        return [input_path]

    elif os.path.isdir(input_path):
        image_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if Path(f).suffix.lower() in supported_formats
        ]

        if not image_files:
            logger.error(f"No supported images found in: {input_path}")
            logger.info(f"Supported formats: {', '.join(supported_formats)}")
            sys.exit(1)

        return sorted(image_files)

    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)


def main():
    """Main application entry point."""
    args = parse_arguments()

    # Print banner
    print_banner()

    # Setup logger
    logger = Logger(verbose=args.verbose)

    # Validate input
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)

    # Setup output directory
    if args.output is None:
        if os.path.isdir(args.input):
            args.output = os.path.join(args.input, "image_upscale")
        else:
            args.output = os.path.join(os.path.dirname(args.input), "image_upscale")

    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Output directory: {args.output}")

    if args.format:
        logger.info(f"Output format: {args.format.upper()}")

    # Get image files
    image_files = get_image_files(args.input, logger)
    logger.info(f"Found {len(image_files)} image(s) to process")

    # Initialize components
    client = ComfyUIClient(host=args.host, port=args.port, logger=logger)
    converter = WorkflowConverter(logger)
    processor = ImageProcessor(client, converter, logger)
    workflow = get_default_workflow()

    # Process images
    print()
    successful = 0
    failed = 0
    failed_files = []

    with tqdm(
        total=len(image_files),
        desc="Overall progress",
        unit="image",
        position=0,
        leave=True,
        ncols=80,
    ) as overall_pbar:
        for image_path in image_files:
            try:
                processor.process(
                    workflow, image_path, args.output, args.prefix, args.format
                )
                successful += 1
            except Exception as e:
                logger.error(
                    f"Failed to process {os.path.basename(image_path)}: {str(e)}"
                )
                failed += 1
                failed_files.append(os.path.basename(image_path))
            finally:
                overall_pbar.update(1)

    # Summary
    print()
    print("=" * 60)
    logger.success("Processing complete!")
    logger.info(f"Successful: {successful}, Failed: {failed}")

    if failed_files:
        logger.warning("Failed files:")
        for filename in failed_files:
            print(f"   • {filename}")

    logger.info(f"Output saved to: {args.output}")
    print("=" * 60)

    # Cleanup
    client.close()


if __name__ == "__main__":
    main()
