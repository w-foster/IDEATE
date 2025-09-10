import base64
from pathlib import Path
from typing import List
from langchain_core.messages import HumanMessage 

from langgraphs.utils import encode_image_to_data_url

NOVELTY_CHECK_PROMPT = """
You are an agent in a wider system whose goal is to promote creative exploration/thinking in generative models, in order to better meet the design task specified by a user.
You are the Novelty Checker agent. Your role is to check if a new image is novel/creative/interesting enough, relative to the existing archive of images.

You will be given:
- the new image
- a labelled list of the existing images, already in the archive
- the user's design task, just for context

You need to very carefully compare the new image to each of the existing images, and check whether or not they are sufficiently semantically similar.
For something to be sufficiently semantically similar, this must mean that the new image is NOT a creative/novel/interesting addition to the archive -- perhaps the new image shares too many central/key features with the existing image.
EXAMPLE: if the user's design task is 'a mythical creature', and an existing image is a purple dragon, then another purple dragon may not be a creative addition to the archive, UNLESS it meaningfully differs in other ways.
Consider what counts as meaningful difference -- this may be composition, such as subject scale for tasks whose subject cannot differ too much, whereas for others the composition may be less relevant and novelty may require difference elsehwere (e.g. in the mythical creature example this is likely the case).

Remember -- our goal is to promote creative exploration; BUT, you must be careful not to hinder this by suggesting that someting is too semantically similar when it is in fact creative -- you MUST strike a balance.
"""



def build_novelty_human_message(
    new_img_path: str,
    archive_img_paths: List[str],
    design_task: str
) -> HumanMessage:
    """
    Constructs a HumanMessage that includes the new image followed by all archive images,
    then asks which (if any) archive image is most semantically similar to the new one.
    """
    content = []
    content.append({"type": "text", "text": f"FOR CONTEXT, my design task is: {design_task}"})

    # New candidate image
    new_data_url = encode_image_to_data_url(new_img_path)
    content.append({"type": "text", "text": "This is the NEW candidate image."})
    content.append({"type": "image_url", "image_url": {"url": new_data_url}})

    # All archive images
    for idx, path in enumerate(archive_img_paths, start=1):
        data_url = encode_image_to_data_url(path)
        content.append({"type": "text", "text": f"\n\nArchive Candidate #{idx}"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})
        

    # Final question
    content.append({
        "type": "text",
        "text": (
            f"Analyse the Archive Candidates #1 through #{len(archive_img_paths)}. Remember, DO NOT FORCE THERE TO BE ONE THAT IS SEMANTICALLY SIMILAR -- that is not what we are hoping for; we are hoping for true accuracy. It is CRITICAL that you are as accurate and objective as possible, and that you value novelty and creativity instead of forcing a semantic similarity claim based on small similarities."
        )
    })

    return HumanMessage(content=content)
