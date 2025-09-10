from typing import List, Tuple, Optional
import base64
from pathlib import Path
from langchain_core.messages import HumanMessage 

from langgraphs.utils import encode_image_to_data_url
from core.branch_context import BranchContext


EVALUATION_PROMPT = """**OVERVIEW**
You are an agent in a wider system whose goal is to promote creative exploration/thinking in generative models, in order to better meet the design task specified by a user.
You are the Evaluator agent. Your role is to perform a pairwise evaluation -- you must compare two images, without bias, and select which one you think is best.

**TASK**
You are evaluating a given image according to the following criteria:
- How well does the image align with the user's design task
- How high quality is the image (e.g., hows the image quality? how's the consistency within the image? is there anything genuinely erroneous?)
- How creative, novel, or interesting is the image

**GUIDANCE**
- Given the domain we are in (of which you will receive a description), try to select the best image, on behalf of the human user. 
- The images will likely be similar in some regards, so try to focus on what is DIFFERENT between the two -- different concepts/features? different quality execution of a shared concept/feature? different degree of interestingness/creativity? different degree of adherence to the design task?
- You shouldn't just pick the image which complies most with the prompt/design task -- the purpose of the system is to promote divergent exploration of the design space for this design task. But at the same time, task-alignment is very important. You need to be careful about where to draw the line on extra features.
-- For example: for the task "a red house", you shouldn't just pick images where the only key feature is a red house, and not much else, just because MORE of the image is a red house --> a different image which has a red house WITHIN some other context may well be more interesting. BUT, this extra context could also cross the line into task misalignment, e.g. if additions that are going too off track are made.

You must try and take a balanced approach -- an image that is slightly less adherent to the design task, but far higher quality, could be better. But that's your call to make. Think clearly and deeply -- this decision is very important.
"""


def create_evaluation_system_prompt(is_convergence_branch: bool) -> str:
    prompt = EVALUATION_PROMPT
    if is_convergence_branch:
        prompt += """

====== IMPORTANT ADDITIONAL INFORMATION ======
You are operating in a CONVERGENT BRANCH of this creative session. The user provided their INITIAL/OVERALL design task earlier, and has now created a NEW DESIGN TASK for this branch by selecting one or more reference images and adding extra text.

When evaluating, prioritize alignment with the user's CURRENT intent as clarified by this branch context. Consider the initial design task as background, but treat the branch text and selected reference images (and their underlying ideas) as the immediate specification of what “task-aligned” means here.

In general, judge in favor of the CURRENT branch intent (not generic novelty or the older task interpretation), while still valuing creativity and image quality.
You should begin by considering WHAT the current design task actually is, based on the original task and interactivity payloads.

**ADDITIONAL INPUTS**
- Alongside the initial design task (text-only), you will be given the MOST RECENT interactivity payload (selected images + text provided by the user)
- BELOW this, you will be given a list of ALL PRIOR interactivity payloads, to help you understand the full context of the most recent one. BUT, remember the MOST RECENT one (which we show first) is most important and relates to the user's current intent.

**IMPORTANT GUIDANCE FOR TASK-ALIGNMENT**
- In the case of two or more interactivity payloads (on top of the OG design task), intent which is contained within prior interactivity payloads MIGHT STILL BE RELEVANT
-- For example, suppose the task is "a dog", and then an interactivity payload consists of a selected image of a very LARGE dog, and inputted text that says "the big ones are cool!". 
-- Then suppose there's the CURRENT interactivity payload where they select a black dog (in the archive created ideally of BIG dogs) and say "black is what I want" — then we must assume their new design task is not just ANY black dog but a *BIG* black dog!
-- The only case you should not take the prior interactivity payload into account in this way is if it is clear that the new one invalidates the previous one.
====== END OF IMPORTANT ADDITIONAL INFORMATION ======
"""
    return prompt


def build_evaluation_human_message(
    img_file_names: Tuple[str, str], 
    design_task: str, 
    domain_description: str,
    flip_order: bool = False,
    branch_context: Optional[BranchContext] = None
) -> HumanMessage:
    
    content = []
    if branch_context is None:
        content.append({"type": "text", "text": f"FOR CONTEXT, my design task is: {design_task}\n\nAnd a description of the domain we are operating in is:\n{domain_description}"})
    else:
        content.append({"type": "text", "text": f"""
FOR CONTEXT, here is my INITIAL/OVERALL design task:
"{design_task}"

And here is the NEW design task for this branch:
"{branch_context.new_design_task}"
"""})
        for idx, path in enumerate(branch_context.reference_img_paths, start=1):
            content.append({"type": "text", "text": f"\n\nSelected Image #{idx}"})
            content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(path)}})
        if branch_context.reference_ideas:
            content.append({"type": "text", "text": "And here are the IDEA(S) behind those image(s):"})
            for idx, idea in enumerate(branch_context.reference_ideas, start=1):
                content.append({"type": "text", "text": f"\n\nSelected Image IDEA #{idx}\n```\n{idea}\n```\n\n"})

        # If prior payloads exist, render them separately
        if getattr(branch_context, "prior_reference_img_paths", None) or getattr(branch_context, "prior_branch_texts", None):
            content.append({"type": "text", "text": "\n\n====== PREVIOUS BRANCH CONTEXT ======"})
            if branch_context.prior_branch_texts:
                for i, t in enumerate(branch_context.prior_branch_texts, start=1):
                    content.append({"type": "text", "text": f"\nPrior Branch Text #{i}:\n```\n{t}\n```\n"})
            if branch_context.prior_reference_img_paths:
                for idx, path in enumerate(branch_context.prior_reference_img_paths, start=1):
                    content.append({"type": "text", "text": f"\n\nPrior Selected Image #{idx}"})
                    content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(path)}})
            if branch_context.prior_reference_ideas:
                content.append({"type": "text", "text": "And here are the IDEA(S) behind the prior image(s):"})
                for idx, idea in enumerate(branch_context.prior_reference_ideas, start=1):
                    content.append({"type": "text", "text": f"\n\nPrior Selected Image IDEA #{idx}\n```\n{idea}\n```\n\n"})

    if not flip_order:
        img1 = encode_image_to_data_url(img_file_names[0])
        img2 = encode_image_to_data_url(img_file_names[1])
    else:
        img2 = encode_image_to_data_url(img_file_names[0])
        img1 = encode_image_to_data_url(img_file_names[1])

    content.append({"type": "text", "text": "Image 1:"})
    content.append({"type": "image_url", "image_url": {"url": img1}})

    content.append({"type": "text", "text": "Image 2:"})
    content.append({"type": "image_url", "image_url": {"url": img2}})
        
    # Final question
    content.append({
        "type": "text",
        "text": "Evaluate the images carefully, and do so on behalf of the user. This is an important task -- try to think like a human; take everything into account; take the design task into account too, and also the domain as context."
    })

    return HumanMessage(content=content)
