
import argparse
import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


MODEL = "gpt-5.6-sol"


def image_to_data_url(image_path):
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/jpeg"

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate product descriptions and marketing slogans from product images."
    )

    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="One or more product image files",
    )

    parser.add_argument(
        "--info",
        default="",
        help="Optional additional product information",
    )

    args = parser.parse_args()

    client = OpenAI()

    instructions = """
Analyze the supplied product images and create marketing content.

Important rules:
- Clearly distinguish information visible in the images from facts supplied by the user.
- Do not invent materials, dimensions, technical specifications, certifications,
  performance claims, brand information, or other facts that cannot be determined
  from the images or the user's text.
- If something is uncertain, say that it cannot be confirmed.

Return the answer using exactly these sections:

VISIBLE IN THE IMAGES:
Briefly list relevant observations from the images.

USER-PROVIDED FACTS:
List the facts supplied by the user. If none were supplied, say "None provided."

PRODUCT DESCRIPTION:
Write a clear and useful marketing description based only on supported information.

MARKETING SLOGANS:
Write 3 short marketing slogans.
"""

    content = [
        {
            "type": "input_text",
            "text": instructions + "\n\nUser-provided product information:\n" + args.info,
        }
    ]

    for image_path in args.images:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(image_path),
            }
        )

    try:
        response = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        print("\n" + "=" * 60)
        print("PRODUCT MARKETING GENERATOR")
        print("=" * 60 + "\n")
        print(response.output_text)

    except Exception as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()