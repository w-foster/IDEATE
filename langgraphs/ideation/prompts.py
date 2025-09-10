from typing import List, Optional

from core.branch_context import BranchContext

from langchain_core.messages import HumanMessage

from langgraphs.utils import encode_image_to_data_url

def create_convergence_extra_prompt() -> str:
    return f"""====== IMPORTANT ADDITIONAL INFORMATION ======
You are operating in a CONVERGENT BRANCH of the current creative session. What this means is, we are no longer in the initial fully divergent phase, where the user has only given their design task.
Rather, a whole archive of initial, divergent images have been generated, based on that initial design task and a generated creative strategy. And now, the user has given a NEW DESIGN TASK, and selected one or more images from the archive.

This new text-based design task, plus the selected images, together form the design task that the system is now working with. Importantly, the approach should remain largely the same -- we still want to promote divergence / creativity as much as possible, WITHIN THE CONSTRAINTS of the design task (i.e., task aligment is imperative).

Alongside the original/initial design task (and the other inputs), you will be given the new design task (the text input and the images the user selected), for additional context. However, be sure to still follow the creative strategy very closely (just note that you may not be given any seed ideas).

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

def create_ideation_system_prompt(is_convergence_branch: bool) -> str:
    prompt = """**OVERVIEW AND ROLE**
You are an agent in a system whose goal is to promote creative/novel/interesting yet high-quality and task-aligned outputs from generative models (in this case, image generation via diffusion).
You are the Ideator agent. Your role is to follow a creative strategy to generate novel, interesting and creative IDEAS, which will later be turned into PROMPTS by another agent (not you).

**TASK AND END GOAL**
You will be given the following:
- A description of the domain for the task
- The user's design task
- A creative strategy
- The existing ideas in the archive
- One or more SEED ideas

Your task/goal:
- Your task is to take the seed ideas, apply the creative thinking strategy in detail, and in doing so generate ONE new idea. 
- Importantly, your role is NOT to generate the prompt -- you are NOT generating a prompt etc. You are generating the IDEA ITSELF, which should take the form of a paragraph description outlining your idea, or a similar prose-based outline of your idea.
- The purpose of having your role and a SEPARATE prompt engineering role, is that you can SOLELY focus on ideation, and creating semantically interesting ideas which will then LATER be refined down into prompts for the diffusion model itself. As such, you should not worry about prompt engineering, and focus solely on ideation.

**GUIDANCE ON IN-CONTEXT IDEATION**:
- Part of your input will be the existing ideas in the archive -- ones that have already been generated as part of this creative process. Use this to your advantage when you are exploring interestingness/novelty, since it will help to ground you in the sorts of ideas that might have already been explored.
- Additionally, though, and VERY IMPORTANTLY: you do not necessarily NEED to be wildly novel in your ideas -- simple ideas CAN be more performant, and so if you recognise that the archive has not explored the more obvious ideas for this design task, perhaps because the Ideator instance that came before you was going for more wild novelty/exploration of the design space, then *FEEL FREE* to keep it more simple. 
- **IMPORTANT**: What matters is that we produce an OVERALL interesting set of images; it is still desirable for this set to contain some more simple, less wild ideas -- there may be some 'free wins' here, especially when the archive is low capacity. For example, if we are exploring the design task of "a boat", and there aren't many ideas yet, you may want to try to create a high-quality yet SIMPLE idea -- speedboat by the beach, or a canoe in a lake, or an old pirate ship -- rather than going for wildly novel ideas for the sake of it.

**GUARDRAILS**
- You **MUST** follow any guardrails which are explicitly outlined in the creative strategy you are given, especially when they relate to realism vs non-realism.
- For example, if the strategy emphasises realism, you should avoid abstract/sci-fi/fantastical ideas -- stay grounded. There is a lot of interestingness/creativity/novelty/beauty to explore in realism too, without descending into absurdity.

- Separate side note: in the past, some ideas have been strangely ABOUT exploration/creativity -- this is missing the point. The idea is not (necessarily) to be ABOUT exploration/creativity, you are USING creative thinking to explore and generate more interesting ideas, regardless of what those ideas are about.

After you are given the necessary components for your task, go ahead and begin -- rememeber, output ONE idea only, and MAKE SURE to use the SEED ideas when applying your creative strategy, as inspiration or sources for variation / mutation. Feel free to think deeply, just make sure your very last output matches the schema you will be provided.


"""
    if is_convergence_branch:
        prompt += f"\n\n{create_convergence_extra_prompt()}"
    return prompt



def create_ideas_string(ideas: List[str], idea_type: str) -> str:
    out = ""
    for idx, idea in enumerate(ideas):
        out += f"{idea_type} {idx + 1}:\n```\n{idea}\n```\n\n"
    return out 



def create_user_ideation_prompt(creative_strategy: str, design_task: str, domain_description: str, seed_ideas: List[str], archive_ideas_except_seeds: List[str], branch_context: Optional[BranchContext]) -> HumanMessage:
    seed_ideas_str = create_ideas_string(ideas=seed_ideas, idea_type="Seed Idea")
    archive_ideas_except_seeds_str = create_ideas_string(ideas=archive_ideas_except_seeds, idea_type="Existing Idea")
    
    content = []
    content.append({"type": "text", "text": f"""
Here's a description of the domain that I'm exploring in:
```
{domain_description}
```
"""})

    if branch_context is None:
        content.append({"type": "text", "text": f"""
And here's MY design task -- this is what I want the system to help me create/achieve:
```
{design_task}
```
"""})
    else:
        content.append({"type": "text", "text": f"""
And here's my INITIAL/OVERALL design task -- this is what I want the system to help me create/achieve in general:
```
{design_task}
```
"""})
        content.append({"type": "text", "text": f"""
And here is the NEW design task for this branch, along with the image(s) I selected:
```
{branch_context.new_design_task}
```
"""})
        for idx, path in enumerate(branch_context.reference_img_paths, start=1):
            data_url = encode_image_to_data_url(path)
            content.append({"type": "text",      "text": f"\n\nSelected Image #{idx}"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        
        content.append({"type": "text", "text": "And here are the IDEA(S) behind those image(s):"})
        for idx, idea in enumerate(branch_context.reference_ideas, start=1):
            data_url = encode_image_to_data_url(path)
            content.append({"type": "text",      "text": f"\n\nSelected Image IDEA #{idx}\n```\n{idea}\n```\n\n"})

        # If prior payloads exist, render them separately
        if getattr(branch_context, "prior_reference_img_paths", None) or getattr(branch_context, "prior_branch_texts", None):
            content.append({"type": "text", "text": "\n\n====== PREVIOUS BRANCH CONTEXT ======"})
            if branch_context.prior_branch_texts:
                for i, t in enumerate(branch_context.prior_branch_texts, start=1):
                    content.append({"type": "text", "text": f"\nPrior Branch Text #{i}:\n```\n{t}\n```\n"})
            if branch_context.prior_reference_img_paths:
                for idx, path in enumerate(branch_context.prior_reference_img_paths, start=1):
                    data_url = encode_image_to_data_url(path)
                    content.append({"type": "text",      "text": f"\n\nPrior Selected Image #{idx}"})
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
            if branch_context.prior_reference_ideas:
                content.append({"type": "text", "text": "And here are the IDEA(S) behind the prior image(s):"})
                for idx, idea in enumerate(branch_context.prior_reference_ideas, start=1):
                    content.append({"type": "text",      "text": f"\n\nPrior Selected Image IDEA #{idx}\n```\n{idea}\n```\n\n"})

    content.append({"type": "text", "text": f"""
What follows are all the EXISTING IDEAS in the Archive (excluding the SEED ideas, which you will be given shortly). You are being given this as context to help you be truly creative and not repetetive -- it should give you an understanding of what's been explored so far in this process. Along with the creative strategy, and the seed ideas as inspiration (launchpad for mutation/variation/new ideas), you should carefully assess what sort of ideas have ALREADY BEEN EXPLORED in this list.
As you go through the creative strategy, step-by-step, be mindful of what routes might have been taken by previous Ideators to produce these existing ideas, and try to be creative and fresh. Perhaps the creative strategy offers guidance towards routes which are yet to be taken.
EXISTING IDEAS IN THE ARCHIVE:
=====
{archive_ideas_except_seeds_str}
=====


What follows next is the entire creative strategy, which you are to use, along with the seed ideas, to generate ONE NEW IDEA. Use the seed ideas as points of inspiration / sources for variation/mutation alongside the creative strategy.
CREATIVE STRATEGY:
```
{creative_strategy}
```

SEED IDEAS:
=====
{seed_ideas_str}
=====

Your role is extremely important, and getting truly creative/novel/interesting outputs is extremely important to me; and so is achieving my design task, and not straying too far from its central spirit.
It's critical that you use the SEED ideas and the CREATIVE STRATEGY to guide your ideation process. You are acting sort of like a mutation operator in this creative exploration, so that we can explore the design space in a better way. Be creative!
"""})

    return HumanMessage(content=content)