import base64
from pathlib import Path
from typing import List
from langchain_core.messages import HumanMessage 

from langgraphs.utils import encode_image_to_data_url


FIND_MOST_SIMILAR_IMAGE_PROMPT = """
You are an agent in a wider system whose goal is to promote creative exploration/thinking in generative models, in order to better meet the design task specified by a user.
The current sub-process involves making a set of comparisons between a newly generated image, and existing images in the archive, in order to find the archive image which the new image is most similar to.

Your role is to compare the new image to a subset of the archive, in order to identify which of the archive images you are presented with is MOST SIMILAR to the new image / vice versa.

You will be given:
- the new image
- a labelled list of the archive images
- the user's design task and the domain description, for context

You need to very carefully compare the new image to each of the existing images, and check the degree of semantic (and stylistic, compositional, etc.) similarity.
It is important that you consider what counts as meaningful differences in images, given the current design task. This may help you to identify what counts as meaningfully similar/different.
For example, if the design task involves a lot of creativity inherently, then two images strongly sharing composition or camera FOV, but with very different subjects, may be less similar than a semantically similar subject presented in compositionally different ways. On the other hand, achieving a novel set of images in more grounded tasks may perhaps demand a wider range of composition / camera location / setting / background details etc.

Ultimately, it is up to you to determine the best approach here, and to compare the new image to each of the archive images VERY carefully.

Think deeply. Once you are done, you can specify your very final output according to the schema you will be provided with.
"""


def build_find_most_similar_img_human_message(
    new_img_path: str,
    domain_description: str,
    selected_archive_img_paths: List[str],
    design_task: str 
) -> HumanMessage:
    content = []
    content.append({"type": "text", "text": f"""
For context, my design task is:
"{design_task}"

And the domain description is:
"{domain_description}"
"""
    })

    # New candidate image
    new_data_url = encode_image_to_data_url(new_img_path)
    content.append({"type": "text", "text": "This is the NEW candidate image."})
    content.append({"type": "image_url", "image_url": {"url": new_data_url}})

    # Selected archive images
    content.append({"type": "text", "text": "\nWhat follows are the selected subset of archive images."})
    for idx, path in enumerate(selected_archive_img_paths, start=1):
        data_url = encode_image_to_data_url(path)
        content.append({"type": "text", "text": f"\n\nArchive Image #{idx}"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    return HumanMessage(content=content)

