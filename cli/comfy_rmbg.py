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
                if input_name and link_id is not None and link_id in link_map:
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

        if node_type == "LoadImage":
            inputs["image"] = (
                uploaded_filename if uploaded_filename else node["widgets_values"][0]
            )
            inputs["upload"] = "image"
            self.logger.verbose(f"LoadImage configured with: {inputs['image']}")

        elif node_type == "LayerMask: LoadBiRefNetModelV2":
            model_name = node["widgets_values"][0]
            inputs["model_name"] = model_name
            inputs["version"] = model_name
            self.logger.verbose(f"BiRefNet model configured: {model_name}")

        elif node_type == "LayerMask: BiRefNetUltraV2":
            param_names = [
                "detail_method",
                "detail_erode",
                "detail_dilate",
                "black_point",
                "white_point",
                "process_detail",
                "device",
                "max_megapixels",
            ]
            for i, val in enumerate(node["widgets_values"]):
                if i < len(param_names):
                    inputs[param_names[i]] = val
            self.logger.verbose("BiRefNetUltraV2 parameters configured", inputs)

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
            self.logger.error(f"Failed to get image: {str(e)}")
            return None

    def get_image_from_url(self, url: str) -> Optional[requests.Response]:
        """Retrieve an image from a URL."""
        try:
            self.logger.debug(f"Fetching image from URL: {url}")
            response = requests.get(url)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get image from URL: {str(e)}")
            return None

    def wait_for_completion(self):
        """Wait for workflow execution to complete with progress bar."""
        with tqdm(
            total=100,
            desc="Processing image",
            unit="%",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
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
        self, workflow_data: Dict, input_path: str, output_dir: str, prefix: str = ""
    ) -> str:
        """Process a single image through the workflow."""
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
        time.sleep(1)  # Brief pause for file system

        # Get preview node
        preview_node_id = self._find_preview_node(workflow_api)

        # Retrieve output
        output_image = self._retrieve_output(prompt_id, preview_node_id)
        if not output_image:
            self.logger.error("Could not retrieve output image")
            sys.exit(1)

        # Save output
        output_path = self._save_output(output_image, input_path, output_dir, prefix)

        # Cleanup
        self._cleanup_input_folder()

        return output_path

    def _find_preview_node(self, workflow_api: Dict) -> str:
        """Find the PreviewImage node ID in the workflow."""
        for node_id, node in workflow_api.items():
            if node["class_type"] == "PreviewImage":
                self.logger.debug(f"Preview node found: {node_id}")
                return node_id

        self.logger.error("No PreviewImage node found in workflow")
        sys.exit(1)

    def _retrieve_output(
        self, prompt_id: str, preview_node_id: str
    ) -> Optional[requests.Response]:
        """Retrieve output image from execution history."""
        spinner = Halo(text="Retrieving output", spinner="dots")
        spinner.start()

        prompt_data = self.client.get_history(prompt_id)
        if not prompt_data:
            spinner.fail("Failed to retrieve output")
            return None

        outputs = prompt_data.get("outputs", {})
        node_output = outputs.get(preview_node_id, {})
        images = node_output.get("images", [])

        if not images:
            spinner.fail("No output images found")
            self.logger.warning("No images in node output")
            return None

        img_info = images[-1]
        if isinstance(img_info, dict):
            filename = img_info["filename"]
            subfolder = img_info.get("subfolder", "")
            folder_type = img_info.get("type", "temp")

            self.logger.debug(
                f"Output image: {filename} (subfolder: {subfolder}, type: {folder_type})"
            )

            output_image = self.client.get_image(filename, subfolder, folder_type)
            if output_image and output_image.status_code == 200:
                spinner.succeed(f"Retrieved output: {filename}")
                return output_image

        spinner.fail("Failed to retrieve output")
        return None

    def _save_output(
        self,
        output_image: requests.Response,
        input_path: str,
        output_dir: str,
        prefix: str,
    ) -> str:
        """Save the output image."""
        base_filename = Path(input_path).stem
        output_filename = f"{prefix}{base_filename}.png"
        output_path = os.path.join(output_dir, output_filename)

        spinner = Halo(text=f"Saving {output_filename}", spinner="dots")
        spinner.start()

        try:
            img = Image.open(io.BytesIO(output_image.content))
            img.save(output_path, format="PNG")
            spinner.succeed(f"Saved: {output_filename}")
            self.logger.debug(f"Output saved to: {output_path}")
            return output_path
        except Exception as e:
            spinner.fail("Failed to save output")
            self.logger.error(f"Save error: {str(e)}")
            sys.exit(1)

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
    """Return the default BiRefNet background removal workflow."""
    return {
        "last_node_id": 14,
        "last_link_id": 10,
        "nodes": [
            {
                "id": 12,
                "type": "LoadImage",
                "pos": [50, 50],
                "size": [315, 314],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [9]},
                    {"name": "MASK", "type": "MASK", "links": None},
                ],
                "properties": {"Node name for S&R": "LoadImage"},
                "widgets_values": ["boy.png", "image"],
            },
            {
                "id": 6,
                "type": "LayerMask: LoadBiRefNetModelV2",
                "pos": [50, 400],
                "size": [315, 58],
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "birefnet_model", "type": "BIREFNET_MODEL", "links": [1]}
                ],
                "properties": {"Node name for S&R": "LayerMask: LoadBiRefNetModelV2"},
                "widgets_values": ["RMBG-2.0"],
            },
            {
                "id": 5,
                "type": "LayerMask: BiRefNetUltraV2",
                "pos": [400, 50],
                "size": [315, 246],
                "flags": {},
                "order": 2,
                "mode": 0,
                "inputs": [
                    {"name": "image", "type": "IMAGE", "link": 9},
                    {"name": "birefnet_model", "type": "BIREFNET_MODEL", "link": 1},
                ],
                "outputs": [
                    {"name": "image", "type": "IMAGE", "links": [4], "slot_index": 0},
                    {"name": "mask", "type": "MASK", "links": [], "slot_index": 1},
                ],
                "properties": {"Node name for S&R": "LayerMask: BiRefNetUltraV2"},
                "widgets_values": ["VITMatte", 4, 2, 0.15, 0.99, False, "cuda", 2],
            },
            {
                "id": 8,
                "type": "PreviewImage",
                "pos": [750, 50],
                "size": [315, 252],
                "flags": {},
                "order": 3,
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 4}],
                "outputs": [],
                "properties": {"Node name for S&R": "PreviewImage"},
                "widgets_values": [],
            },
        ],
        "links": [
            [1, 6, 0, 5, 1, "BIREFNET_MODEL"],
            [4, 5, 0, 8, 0, "IMAGE"],
            [9, 12, 0, 5, 0, "IMAGE"],
        ],
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def print_banner():
    """Print application banner."""
    banner = figlet_format("BG Remover", font="slant")
    print(banner)
    print("=" * 60)
    print("  ComfyUI Background Removal Tool")
    print("  Powered by BiRefNet")
    print("=" * 60)
    print()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove backgrounds from images using ComfyUI and BiRefNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i image.png                    # Process single image
  %(prog)s -i ./images -o ./output        # Process directory
  %(prog)s -i ./images --prefix "nobg_"   # Add prefix to outputs
  %(prog)s -i image.png -v                # Verbose mode
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
        help="Output directory (default: <input>/bg_removed)",
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
            args.output = os.path.join(args.input, "bg_removed")
        else:
            args.output = os.path.join(os.path.dirname(args.input), "bg_removed")

    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Output directory: {args.output}")

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

    with tqdm(
        total=len(image_files),
        desc="Overall progress",
        unit="image",
        position=1,
        leave=True,
    ) as overall_pbar:
        for image_path in image_files:
            try:
                processor.process(workflow, image_path, args.output, args.prefix)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                if logger._verbose:
                    import traceback
                    traceback.print_exc()
                failed += 1
            finally:
                overall_pbar.update(1)

    # Summary
    print()
    print("=" * 60)
    logger.success("Processing complete!")
    logger.info(f"Successful: {successful}, Failed: {failed}")
    logger.info(f"Output saved to: {args.output}")
    print("=" * 60)

    # Cleanup
    client.close()


if __name__ == "__main__":
    main()
