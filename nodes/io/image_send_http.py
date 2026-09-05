"""POST a batch of images to an HTTP endpoint."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import deps, log
from ...modules.compat.types import DICT

REQUIRES = "network"

logger = log.get_logger("nodes.io")

#: Config key that enables this node, named in dependency errors.
FEATURE = "features.network"


class ImageSendHTTP(io.ComfyNode):
    """Upload every image in the batch to ``url`` as a multipart PNG request."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Send HTTP",
            display_name="Image Send HTTP",
            search_aliases=["Image Send HTTP", "upload image", "post image", "webhook"],
            category="WAS Suite/IO",
            description=(
                "Send the images to an HTTP endpoint as a multipart upload. This node "
                "makes an outbound request with the image data, so it is only loaded "
                "when features.network is enabled."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to upload. Every image in the batch is encoded as a PNG "
                        "and sent in the same request, named 'image_0.png', 'image_1.png' and "
                        "so on."
                    ),
                ),
                io.String.Input(
                    "url",
                    default="example.com",
                    tooltip=(
                        "Full address the images are sent to, e.g. "
                        "'https://example.com/api/upload'. The default is a placeholder and "
                        "has to be replaced."
                    ),
                ),
                io.Combo.Input(
                    "method_type",
                    options=["post", "put", "patch"],
                    default="post",
                    tooltip=(
                        "Which HTTP verb the request uses. `post` is the usual choice for an "
                        "upload; pick `put` or `patch` if the receiving endpoint asks for one "
                        "of those."
                    ),
                ),
                io.String.Input(
                    "request_field_name",
                    default="image",
                    tooltip=(
                        "Name of the form field the files are attached under. It has to match "
                        "whatever the receiving endpoint expects, often 'image' or 'file'."
                    ),
                ),
                DICT.Input(
                    "additional_request_headers",
                    optional=True,
                    tooltip=(
                        "Extra HTTP headers to send, as a dictionary of names to values, an "
                        "'Authorization' entry for an endpoint that needs a token, for "
                        "instance. Disconnected, only the default headers are sent."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="status_code",
                    tooltip=(
                        "The HTTP status the endpoint answered with: 200 or 201 for success, "
                        "401 for a rejected token, 404 for a wrong address, 500 for a fault "
                        "at the far end."
                    ),
                ),
                io.String.Output(
                    display_name="result_text",
                    tooltip=(
                        "The body of the endpoint's reply, as text, often JSON holding an id "
                        "or a link for the uploaded image."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images,
        url="example.com",
        method_type="post",
        request_field_name="image",
        additional_request_headers=None,
    ) -> io.NodeOutput:
        from io import BytesIO

        import numpy as np
        from PIL import Image

        requests = deps.require("requests", feature=FEATURE)

        images_to_send = []
        for index, image in enumerate(images):
            array = 255.0 * image.cpu().numpy()
            encoded = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            buffer = BytesIO()
            encoded.save(buffer, "png")
            buffer.seek(0)
            images_to_send.append(
                (request_field_name, (f"image_{index}.png", buffer, "image/png"))
            )

        request = requests.Request(
            url=url,
            method=method_type.upper(),
            headers=additional_request_headers,
            files=images_to_send,
        )
        with requests.Session() as session:
            response = session.send(request.prepare())
        logger.info("sent %s image(s) to %s, status %s", len(images_to_send), url, response.status_code)
        return io.NodeOutput(response.status_code, response.text)
