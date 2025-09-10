import base64
from pathlib import Path
from typing import List, Optional
from langchain_core.messages import HumanMessage 

from langgraphs.utils import encode_image_to_data_url
from core.branch_context import BranchContext

NOVELTY_CHECK_PROMPT = """
You are an agent in a wider system whose goal is to promote creative exploration/thinking in generative models, in order to better meet the design task specified by a user.
You are the Novelty Checker agent. Your role is to check if a new image is novel/creative/interesting enough, relative to the existing archive of images.

You will be given:
- the new image
- a labelled list of the existing images, already in the archive
- the user's design task, for context

You need to very carefully compare the new image to each of the existing images, and check whether or not they are sufficiently semantically similar.
For something to be sufficiently semantically similar, this must mean that the new image is NOT a creative/novel/interesting addition to the archive -- perhaps the new image shares too many central/key features with the existing image.

EXAMPLE: if the user's design task is 'a mythical creature', and an existing image is a purple dragon, then another purple dragon may not be a creative addition to the archive, UNLESS it meaningfully differs in other ways.
Consider what counts as meaningful difference -- this may be composition, such as subject scale for tasks whose subject cannot differ too much, whereas for others the composition may be less relevant and novelty may require difference elsehwere (e.g. in the mythical creature example this is likely the case).
To give another example, if the design task involves a lot of creativity inherently, then two images strongly sharing composition or camera FOV, but with very different subjects, may be sufficiently semantically different. On the other hand, achieving a novel set of images in more grounded tasks may perhaps demand a wider range of composition / camera location / setting / background details etc.

Ultimately, it is up to you to figure out how best to appraoch this -- that is just some generic guidance.
Remember -- our goal is to promote creative exploration; BUT, you must be careful not to hinder this by suggesting that someting is too semantically similar when it is in fact creative -- you MUST strike a balance.
"""


def create_novelty_system_prompt(is_convergence_branch: bool) -> str:
    prompt = NOVELTY_CHECK_PROMPT
    if is_convergence_branch:
        prompt += """

====== IMPORTANT ADDITIONAL INFORMATION ======
You are operating in a CONVERGENT BRANCH of this creative session. This means, the user has given their initial design task, and has since given a NEW INTERACTION, which changes the design task (although only implicitly).

You **MUST** first determine what you think the CURRENT true design task of the user is before thinking about what similarity means for this design task; use the context to do so, and remember that the user's MOST RECENT intent takes priority.
-- That being said, unless the new payload explicitly or implcitly invalidates a previous preference or statement, you should also include prior preferences etc. in the current design task conceptualisation.

Alongside the initial design task, you will be given the MOST RECENT interactivity payload first (selected images + text), followed by any PRIOR interactivity payloads.
====== END OF IMPORTANT ADDITIONAL INFORMATION ======
"""
    return prompt



def build_novelty_human_message(
    new_img_path: str,
    archive_img_paths: List[str],
    design_task: str,
    branch_context: Optional[BranchContext] = None
) -> HumanMessage:
    """
    Constructs a HumanMessage that includes the new image followed by all archive images,
    then asks which (if any) archive image is most semantically similar to the new one.
    """
    content = []
    if branch_context is None:
        content.append({"type": "text", "text": f"FOR CONTEXT, my design task is: {design_task}"})
    else:
        content.append({"type": "text", "text": f"FOR CONTEXT, here is my INITIAL/OVERALL design task:\n\n\"{design_task}\""})
        content.append({"type": "text", "text": f"\n\nHere is the NEW design task for this branch:\n\n\"{branch_context.new_design_task}\""})
        for idx, path in enumerate(branch_context.reference_img_paths, start=1):
            content.append({"type": "text", "text": f"\n\nSelected Image #{idx}"})
            content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(path)}})
        if branch_context.reference_ideas:
            content.append({"type": "text", "text": "And here are the IDEA(S) behind those image(s):"})
            for idx, idea in enumerate(branch_context.reference_ideas, start=1):
                content.append({"type": "text", "text": f"\n\nSelected Image IDEA #{idx}\n```\n{idea}\n```\n\n"})
        if getattr(branch_context, "prior_branch_texts", None) or getattr(branch_context, "prior_reference_img_paths", None):
            content.append({"type": "text", "text": "\n\n====== PREVIOUS BRANCH CONTEXT ======"})
            if branch_context.prior_branch_texts:
                for i, t in enumerate(branch_context.prior_branch_texts, start=1):
                    content.append({"type": "text", "text": f"\nPrior Branch Text #{i}:\n```\n{t}\n```\n"})
            if branch_context.prior_reference_img_paths:
                for idx, path in enumerate(branch_context.prior_reference_img_paths, start=1):
                    content.append({"type": "text", "text": f"\n\nPrior Selected Image #{idx}"})
                    content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(path)}})

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
