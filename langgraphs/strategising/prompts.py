from typing import List, Optional
import base64
from pathlib import Path

from langchain_core.messages import HumanMessage 

from langgraphs.utils import encode_image_to_data_url

from core.branch_context import BranchContext


def create_convergence_extra_prompt() -> str:
    return f"""====== IMPORTANT ADDITIONAL INFORMATION ======
You are operating in a CONVERGENT BRANCH of the current creative session. What this means is, we are no longer in the initial fully divergent phase, where the user has only given their design task.
Rather, a whole archive of initial, divergent images have been generated, based on that initial design task and a generated creative strategy. And now, the user has given a NEW DESIGN TASK, and selected one or more images from the archive.

This new text-based design task, plus the selected images, together form the design task that the system is now working with. Importantly, the approach should remain largely the same -- we still want to promote divergence / creativity as much as possible, WIHTIN THE CONSTRAINTS of the design task (i.e., task aligment is imperative).
However, since there are now reference images along with the textual design task, you need to be careful to generate a strategy which accurately captures the intent of the user; doing so requires a careful analysis of the selected images and the text input they provided.

First, interpret the text + images, to try to determine what sort of exploration is appropriate -- since the convergence dial is now higher, you must be careful to respect the user's intent; if the user outlines a particular change they want, all else should remain as it is, and we should promote divergence / exploration WITHIN THAT CHANGE. If the user is more ambiguous or requests something inherently more open-ended, then it might be more appropriate for wider divergence and exploration to be higher.
Second, you must clearly and carefully capture this in the creative strategy you generate -- you should still be as creative as possbible, use Tavily to research, and promote divergence and creativity within the areas which you recognise to be open for such exploration -- HOWEVER, you must clearly communicate which parts ought to remain as they are, and which parts are free for exploration (but even within those, be clear with the confines / constraints of that exploration). 

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


def create_creative_strategy_generation_system_prompt(is_convergence_branch: bool) -> str:
    prompt = f"""**OVERVIEW AND ROLE**
You are an agent in a system whose goal is to promote creative/novel/interesting yet high-quality and task-aligned outputs from generative models (in this case, image generation via diffusion).
Your role is to create a Creative Strategy, which will be used by an Ideator Agent (another LLM) to create new ideas, from one or more seed ideas. Importantly, these are IDEAS themselves, not the things which bring those ideas to life (i.e. prompts for diffusion).

Create the creative strategy in detail -- it should promote novel, creative, interesting, and diverse thinking. 
- It should promote exploring AROUND the seed ideas, not just simple/boring variations. 
- This strategy will be used, ideally, to explore a creative design space in a divergent fashion.
- It should promote true IDEATION, and bringing in new concepts
- It MUST explicitly focus on creating IDEAS, not e.g. code or prompts -- the IDEAS themselves are what the Ideator will be creating.
- You may mention the domain (image generation via diffusion), but only for the sake of providing context to the scenario.
- It CANNOT BE a concept/idea bank of pre-made ideas from you -- it must promote creation of fresh ideas.

**TASK ALIGNMENT**
- IMPORTANT: the creative strategy you generate must CLEARLY outline that, whilst divergent thinking is promoted, new ideas MUST cohere with the design task specified by the user, that the Ideator will receive.
- For example, if the design task involves generating an image of a human, we should respect that (and not promote swapping it out for a fish or a monkey).
- To aid with this, you will be given some HIGH-LEVEL CONSTRAINTS/GUARDRAILS for the design task, produced by another Agent, which you should refer back to when creating your creative strategy
- Respect the constraints/guardrails for the design task, and incorporate them in your strategy -- for example, if they promote realism over sci-fi or fantasy, you **MUST** make this clear in the strategy that you generate (don't just implicitly avoid including sci-fi promotion; EXPLICITLY mention a realism guardrail)
- However, also be holistic -- the guardrails were generated by another Agent (LLM), and so they may not be perfect. For example, they may overstep the mark in terms of specifying details/creative options that you should consider -- in this case, you are free to ignore those aspects, and your creative strategy needn't be tied down to the examples or ideas generated by the Guardrails Agent

**END GOAL**
As such, your overall goal is this: to generate a creative strategy which is HIGHLY TAILORED to the user's design task (which you will be given), which will then be given to an Ideator to promote them to think AS DIVERGENTLY / INTERESTINGLY / NOVEL as possible, WITHIN THE SCOPE of the design task itself.
Hence, you should think extremely deeply about the specific domain & design task at hand, and how we might best promote creative exploration given that.

You are encouraged to use your Tavily web search tool to gain any information about existing strategies for promoting creative thought in the given domain / design task. Be smart here -- if the design task itself was to, say, generate an image of a new mythical creature, you might explore existing strategies used by creative writers or animators or film directors (etc.), specifically in the domain of coming up with novel mythical creatures. You may well want to then incorporate that into your strategy.

NOTE: When you populate the provided schema for your final output, it should solely contain the strategy. Don't include words like "Based on insights from the Tavily search, this strategy..." -- your final output should be the exact strategy that the Ideator Agent will be given as part of its input.
"""
    if is_convergence_branch:
        prompt += f"\n\n{create_convergence_extra_prompt()}"
    return prompt


def create_user_strategy_request_prompt(domain_description: str, design_task: str, guardrails: str, branch_context: Optional[BranchContext]) -> HumanMessage:
    content = []
    if branch_context is None:
        content.append({"type": "text", "text": f"""
Here's a description of the domain that I'm using:
```
{domain_description}
```

And here's MY design task -- this is what I want the system to help me create/achieve:
```
{design_task}
```

Here are the guardrails generated by another Agent (LLM):
```
{guardrails}
```

Your role is extremely important, and getting truly creative/novel/interesting outputs is extremely important to me; and so is achieving my design task, and not straying too far from its central spirit.
Go ahead and begin.
"""})
    else:
        if branch_context.reference_img_paths is None or len(branch_context.reference_img_paths) == 0:
            raise ValueError("Branch context must have reference images")
        content.append({"type": "text", "text": f"""
Here's a description of the domain that I'm using:
```
{domain_description}
```

And here's my INITIAL/OVERALL design task -- this is what I want the system to help me create/achieve in general:
```
{design_task}
```

And here is the NEW design task for this branch, along with the images I selected:
```
{branch_context.new_design_task}
```
"""})
        for idx, path in enumerate(branch_context.reference_img_paths, start=1):
            data_url = encode_image_to_data_url(path)
            content.append({"type": "text",      "text": f"\n\nSelected Image #{idx}"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        
        content.append({"type": "text", "text": "And here are the IDEAS behind those images:"})
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

        content.append({"type": "text", "text": "\n\nYour role is extremely important, and getting truly creative/novel/interesting outputs is extremely important to me; and so is achieving my design task, and not straying too far from its central spirit. Go ahead and begin."})
    return HumanMessage(content=content)


def create_creative_strategy_refinement_system_prompt(is_convergence_branch: bool = False):
 output = f"""
You are an agent in a system whose goal is to promote creative/novel/interesting yet high-quality and task-aligned outputs from generative model (in this case, image generation via diffusion).
Your role is to REFINE an existing creative strategy, which is being used by an Ideator (LLM) to create new ideas, from one or more seed ideas. Importantly, these are IDEAS themselves, not the prompts which bring those ideas to life by being fed into diffusion.

For context, here is the system prompt that was received by the Creative Strategy Generator:
```
{create_creative_strategy_generation_system_prompt(is_convergence_branch=False)}
```

You will receive some feedback on the current archive of images, and you need to refine the creative strategy accordingly. The way you should do so is as follows:
- Refine the inner contents of the strategy itself. Keep as much as possible the same, but add in / change elements that are necessary to reflect the archive feedback. If it seems as though something WAS already included in the creative strategy, perhaps it was being ignored/missed by the Ideator Agent -- in this case, you should try to make it more explicit & draw more attention to it.
- For example, this might involve adding another axis/lever for variation/creativity, which hasn't been explored sufficiently. Or, it may involve removing an aspect of the strategy which is unintentionally promoting homogeneity. Or it might involve explicitly banning certain variations that have been overdone or that are task-misaligned.

**INMPORTANT**:
- You **MUST** follow the same guidance that is outlined in the system prompt that was provided to the Creative Strategy Generator, since you are essentially outputting a new creative strategy, just like they were -- the only difference is you are doing so by REFINING an existing one.
- For example (but not limited to this), you must make sure to remain task-aligned. This means you **MUST** take the Archive Analysis/Feedback with a grain of salt --> if the analysis comes back with an assessment that the archive is missing something but that thing is outside of the scope of the current user design task, then you should avoid refining the creative strategy to avoid it.
- As such, you should only implement changes that you have vetted to align with the same guidance outlined in the Creative Strategy Generator system prompt, and that generally aligns with the design task at hand.
- You **MUST** follow the guardrails that were also given to the initial creative strategy generator agent. You will be given these shortly.

Think deeply about the changes you want to make. Your job is extremely important in this system. When you're ready to make your final output, be sure to output the ENTIRE REFINED CREATIVE STRATEGY (so this includes the main strategy, and all updates).
"""
 if is_convergence_branch:
  output += "\n\n" + create_convergence_extra_prompt()


def create_user_strategy_refinement_request_prompt(domain_description: str, design_task: str, current_strategy: str, archive_analysis: str, guardrails: str, branch_context: Optional[BranchContext] = None) -> str:
    if branch_context is None:
        return f"""
Here's a description of the domain that I'm using:
```
{domain_description}
```

And here's MY design task -- this is what I want the system to help me create/achieve:
```
{design_task}
```

Here is the current creative strategy in full:
```
{current_strategy}
```

And here is the feedback / archive analysis:
```
{archive_analysis}
```

And here are the GUARDRAILS for creative strategy generation:
```
{guardrails}
```

Your role is extremely important, and getting truly creative/novel/interesting outputs is extremely important to me; and so is achieving my design task, and not straying too far from its central spirit.
Go ahead and begin.
"""
    else:
        images_block = "\n".join([f"Selected Image #{idx}\n" for idx, _ in enumerate(branch_context.reference_img_paths, start=1)])
        ideas_block = "\n".join([f"Selected Image IDEA #{idx}\n```\n{idea}\n```\n" for idx, idea in enumerate(branch_context.reference_ideas, start=1)])
        return f"""
Here's a description of the domain that I'm using:
```
{domain_description}
```

And here's my INITIAL/OVERALL design task -- this is what I want the system to help me create/achieve in general:
```
{design_task}
```

And here is the NEW design task for this branch, along with the images I selected and the ideas behind those images:
```
{branch_context.new_design_task}
```
{images_block}

And here are the IDEAS behind those images:
{ideas_block}

Here is the current creative strategy in full:
```
{current_strategy}
```

And here is the feedback / archive analysis:
```
{archive_analysis}
```

And here are the GUARDRAILS for creative strategy generation:
```
{guardrails}
```

Your role is extremely important, and getting truly creative/novel/interesting outputs is extremely important to me; and so is achieving my design task, and not straying too far from its central spirit.
Go ahead and begin.
"""


"""
You will receive some feedback on the current archive of images, and you need to refine the creative strategy accordingly. The way you should do so is twofold:
(1) Refine the inner contents of the strategy itself. Keep as much as possible the same, but add in / change elements that are necessary to reflect the archive feedback.
For example, this might involve adding another axis/lever for variation/creativity, which hasn't been explored sufficiently. Or, it may involve removing an aspect of the strategy which is unintentionally promoting homogeneity. Or it might involve explicitly banning certain variations that have been overdone or that are task-misaligned.

(2) Add an 'UPDATE N' (if there are no updates yet, 'UPDATE 1') onto the bottom of the strategy. This should act like some current feedback, outlining e.g. the current state of affairs. So for example, this might point out that you added XYZ to the strategy, and that most of the current images fall into value A for that thing, but that B, C, D, etc. are available and that these should be explored too.
Note: DON'T delete previous updates, however you can supercede / override them in your new update by explicitly saying ignore UDPATE N-1 etc., if that's necessary (it may not be).
"""



ARCHIVE_ANALYSIS_PROMPT = """
You will be shown a set of images which are all images generated through a system whose goal is to use LLMs/agents to promote creativity and interesting outputs from diffusion models.
Driving the generation of these images was a range of CREATIVE STRATEGIES, which were generated by an LLM and which set out guidance for how an LLM (in a fresh LLM call) should create a NEW IDEA, given a couple of seed ideas from the existing archive of previously generated ideas. (These ideas are then turned into prompts, which are used to generate images.)
Your task is to evaluate the current creative strategy by looking at the images that have been generated using it so far.

You will be shown ALL the images in the archive, and you will be told which images were generated directly from the CURRENT creative strategy -- these are the most relevant, however you should still look over the entire archive.

Look at all the images carefully, and think deeply: what (if anything) seems to be working well, what (if anything) seems to not be working well; is there anything missing from what the strategy seems to be promoting? Or is it perfect as is.
Some things you might want to look out for are:
- task misalignment (however, you need to be careful here: sometimes this may be an issue with the prompt itself, not the idea, which is therefore not an issue with the creative strategy. So be holistic; only try to identify shortcomings that could clearly be attributed to IDEATION itself)
- common features/characteristics among most of the images (suggestive of homogeneity and lack of variation in that area)
- any positive feedback too, on what seems to be working well in terms of creativity/novelty/variation/interestingness

**IMPORTANT**
- Whilst you are giving a holistic analysis of the archive, your feedback will be used downstream, and so you still have a responsibility to remain ALIGNED to the DESIGN TASK at hand
- As such, DO NOT give feedback about the images that is beyond the scope of the design task -- for example, if the design task was "a cat playing with a dog", don't say that the images are missing a human playing with them; you are operating within the bounds of the design space set out by the design task itself -- this is important.
- You will be given some Gaurdrails, which outline some constraints/guardrails of the design task itself -- you *MUST* keep your feedback within the constraints of these Gaurdrails
-- For example, if the Gaurdrails say to avoid adding humans, don't say "there aren't enough humans"; if it says stick to realism, don't say "there's no fantasy elements"

**ADDITIONAL SPECIFIC RULES**
- Try to *AVOID* saying that the presence of HUMANS, or certain human activity, is missing from the archive; for example, if the design task is "a modern house carved into a mountain", we are focusing on the house itself and the mountainous setting -- just because no images thus far have prominent humans in the photo, does not mean the creative strategy is failing. By default, assume that humans do not need to be present in the images (and only break this rule if it is a clear/obvious possibility of the design task)
- *DO NOT* overstep the mark and give criticism where there is none. It is imperative that you take an objective lens to this -- maximising criticism or feedback is *NOT* the end goal, but spotting issues when you are confident they are there is. If you believe the strategy to be working well based on the state of the archive, *DO NOT* force feedback for the sake of it. It is equally as valid and important for you to say that it's working well. Remember, you can also give a balance of both praise and feedback, if that makes the most sense.

(Remember, be sure to also compare images generated by previous strategies to those generated by the current strategy -- if you see a theme / issue with the old images that IS NO LONGER PRESENT in the recent images generated by the current strategy, then it may be that this issue has already been solved, so bear that in mind.)

Recaps: you won't be shown the strategy itself; your goal is to think deeply about the archive of images themselves. And don't overstep the mark or force feedback where there is none: your role isn't to maximise criticism, your goal is to be as objective as possible. Your answer will have a significant impact on the performance of the system as a whole, and being truthful and objective is the best route to take.
"""

def create_archive_analysis_system_prompt(is_convergence_branch: bool) -> str:
    prompt = ARCHIVE_ANALYSIS_PROMPT
    if is_convergence_branch:
        prompt += """

====== IMPORTANT ADDITIONAL INFORMATION ======
You are operating in a CONVERGENT BRANCH of this creative session. The user supplied a NEW DESIGN TASK for this branch and selected one or more reference images (with associated ideas). Your archive analysis should:
- Evaluate how recent images reflect the CURRENT branch intent, rather than the original text-based design task -- as such, you **MUST** first consider WHAT the current design task actually is, using the original text-based design task plus the interactivity payload (selected imgs, new text) as context.
- Keep feedback within the scope of the CURRENT branch intent and guardrails; avoid recommending directions outside it.
- In general, you might want to be somewhat stricter or more confined, since we have taken at least one step towards convergence if we are here. That being said, we still want to promote divergent exploration WITHIN the confines of the new design task -- but we must respect what the user has said, especially if it is quite explicit or directly outlines a desired change or smth new etc.

**IMPORTANT GUIDANCE FOR TASK-ALIGNMENT**
- In the case of two or more interactivity payloads (on top of the OG design task), intent which is contained within prior interactivity payloads MIGHT STILL BE RELEVANT
-- For example, suppose the task is "a dog", and then an interactivity payload consists of a selected image of a very LARGE dog, and inputted text that says "the big ones are cool!". 
-- Then suppose there's the CURRENT interactivity payload where they select a black dog (in the archive created ideally of BIG dogs) and say "black is what I want" — then we must assume their new design task is not just ANY black dog but a *BIG* black dog!
-- The only case you should not take the prior interactivity payload into account in this way is if it is clear that the new one invalidates the previous one.
====== END OF IMPORTANT ADDITIONAL INFORMATION ======
"""
    return prompt


def build_archive_analysis_human_message(
    design_task: str,
    domain_description: str,
    archive_img_paths: List[str],
    num_offspring: int,
    branch_context: Optional[BranchContext] = None
) -> HumanMessage:
    """
    Constructs a HumanMessage that includes all the images in the archive
    """
    content = []
    if branch_context is None:
        content.append({"type": "text", "text": f"FOR CONTEXT, the design task is:\n```\n{design_task}\n```\n\n"})
        content.append({"type": "text", "text": f"And the domain description is:\n```\n{domain_description}\n```\n\n"})
    else:
        content.append({"type": "text", "text": f"""
FOR CONTEXT, here is my INITIAL/OVERALL design task:
```
{design_task}
```

And here is the NEW design task for this branch, along with the selected images and their ideas.
"""})
        for idx, path in enumerate(branch_context.reference_img_paths, start=1):
            data_url = encode_image_to_data_url(path)
            content.append({"type": "text",      "text": f"\n\nSelected Image #{idx}"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        if branch_context.reference_ideas:
            content.append({"type": "text", "text": "And here are the IDEA(S) behind those image(s):"})
            for idx, idea in enumerate(branch_context.reference_ideas, start=1):
                content.append({"type": "text",      "text": f"\n\nSelected Image IDEA #{idx}\n```\n{idea}\n```\n\n"})
        content.append({"type": "text", "text": f"And the domain description is:\n```\n{domain_description}\n```\n\n"})

    # All archive images (those generated before the current strategy)
    content.append({
        "type": "text",
        "text": "What follows are the archive images. To begin, these are all the images that were generated with strategies BEFORE the current one:\n"
    })
    for idx, path in enumerate(archive_img_paths[:-num_offspring], start=1):
        data_url = encode_image_to_data_url(path)
        content.append({"type": "text",      "text": f"\n\nArchive Image #{idx}"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    # Now show only the last `num_offspring` images (the ones from the CURRENT strategy)
    content.append({
        "type": "text",
        "text": f"And these are the {num_offspring} images generated by the CURRENT strategy:\n"
    })
    for idx, path in enumerate(archive_img_paths[-num_offspring:], start=1):
        data_url = encode_image_to_data_url(path)
        content.append({"type": "text",      "text": f"\n\nArchive Image #{idx}"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})


    # Final call to action
    content.append({"type": "text", "text": "\n\nGo ahead and begin."})

    return HumanMessage(content=content)



def create_convergence_extra_prompt_for_guardrails() -> str:
    return f"""====== IMPORTANT ADDITIONAL INFORMATION ======
You are operating in a CONVERGENT BRANCH of the current creative session. What this means is, we are no longer in the initial fully divergent phase, where the user has only given their design task.
Rather, a whole archive of initial, divergent images have been generated, based on that initial design task and a generated creative strategy. And now, the user has given a NEW DESIGN TASK, and selected one or more images from the archive.

This new text-based design task, plus the selected images, together form the design task that the system is now working with. Importantly, the approach should remain largely the same -- we still want to promote divergence / creativity as much as possible, WITHIN THE CONSTRAINTS of the design task (i.e., task aligment is imperative).
However, since there are now reference images along with the textual design task, you need to be careful to generate guardrails which relate directly to the intent of the user (i.e., in terms of the overall current design task they have in mind); doing so requires a careful analysis of the selected images and the text input they provided.

First, interpret the text + images, to try to determine what the current OVERALL design task of the user is, given the context of their previous/initial design task input.
Then, consider how this might change the constraints/guardrails at hand -- it is possible that these DO NOT CHANGE, but it is also possible that the new input from the user requires a reconsiderations of the guardrails as compared to the initial or previous design tasks.
====== END OF IMPORTANT ADDITIONAL INFORMATION ======
"""


def create_constraints_system_prompt(is_convergence_branch: bool):
    output = """*OVERVIEW*
You are an agent within a wider system whose goal is to promote creativity/novelty/interestingness from the outputs of LLMs, when those outputs are used for downstream creative/generative tasks. In our case, the LLMs/agents are collaborating to produce prompts to then be used in image generation via advanced diffusion models. 

- A key part of the system is the use of a creative strategy, which is generated by a Creative Strategy Generator Agent (LLM), to be tailored to the user's particular design task. 
- For example, if the users task is "a photo of a cow in a field", the creative strategy should provide guidance to the Ideator Agent (LLM) when it's ideating new ideas for that task. 
- As such, it needs to promote divergence exploration AS MUCH AS POSSIBLE, WITHIN THE CONFINES/CONSTRAINTS of the task. That latter point is something you are working on -- TASK ALIGNMENT. 

*YOUR ROLE*
Your role to map out the high level constraints of the current design task, before the creative strategy has been generated. The purpose of this is to promote task alignment, and to avoid promoting diversity at the sacrifice of alignment. 
- You will be given the current design task, and you need to assess it to map out some high level constraints, which will be given to the Creative Strategy Generator Agent (an LLM) to help ground it when it's creating the strategy, and help it to avoid promoting divergence which is misaligned. 
- For example, suppose the design task is somewhat open ended and ambiguous like "a boat" -- here, it is not entirely clear what the bounds of the design space that the user has in mind are. In such a case, WE MUST BE CAUTIOUS, and a general rule is as such: 
-- AVOID INTRODUCING SCI-FI OR FANTASTICAL ELEMENTS BY DEFAULT, UNLESS THEY ARE CLEARLY IMPLICIT IN THE DESIGN TASK. 
-- A consequence of this: promote realism unless the task is explicitly fantastical/abstract/scifi etc.

- To pursue an example further: an LLM might take "a boat" and design a creative strategy which promotes wild and wonderful ideas, such as flying ships, and ships powered by mythical energy, and so on, but this is something we generally want to try and avoid, unless the design task implicitly welcomes this sort of exploration (e.g., if it has clear, implicit sci-fi or fantasy elements, such as robotics or an abstract concept or whatever). We must avoid over-promoting divergence when that divergence would be considered 'absurdity', in relation to the current task.

*OUTPUT AND END GOAL*
- As such, your goal is to produce a small set of constraints or high level guidance for the task. This should not be long, and it should not go into too many details or specifics -- stick with high level concepts. 
*GENERAL OUTPUT RULES*:
- This should be a SMALL SET of points, ideally no more than 2 or 3 short sentences
- It should primarily outline the tone in terms of the appropriateness of realism Vs non-realism -- this should be the first point 
- It should *NOT* go into details on the sorts of things that the Creative Strategy Generator Agent could put in the creative strategy (e.g., it should not provide a list of plausible building materials for "a house" task) -- it should stick to concepts and fundamental guidance 
- Once you feel like you have covered what's most important for the creative strategist to stay aligned, finish there -- do *NOT* put extra points just for the sake of it. 
- You are *NOT* creating the creative strategy itself -- just the guardrails, and your output will be fed to the Creative Strategist, as a small part of their context

*IMPORTANT ADDITIONAL RULES*
- You **MUST NOT** make assumptions or add a constraint that forces the creative strategy in a particular direction. Generally, this means you should not enforce SPECIFIC DETAILS OR IDEAS that are not high-level requirements of the task itself.
- For example, if the task was "a coffee shop logo", it would be WRONG to enforce that there must be a coffee cup present in the logo. But, it would be good to require something which would be considered a logo, rather than, say, a full image painting or a photograph
- If the task was "a painting of two cats playing", it would be WRONG to start listing details about the cats (their breed, the style they are presented, etc.) and also WRONG to list/enforce requirements on the painting style or composition ('centred cats', 'playing on the floor', etc. -- these are NOT necessary requirements of the task at hand)
- It is absolutely imperative that you avoid imposing low-level constraints or specific details or creative ideas or directions within your final output. You **MUST** stick to high level concepts, such as:
-- Should we stick to realism rather than sci-fi or fantasy?
-- What are the high-level, essential objects or subjects or relations that should hold? (not in detail about HOW these things should hold, unless explicitly stated)
-- Should we permit bringing in additional subjects, such as humans? (generally avoid permitting this; generally, explicitly disallow)
-- Are there any high-level constraints on the STYLE (e.g., 'painting', 'photorealism', 'vector art')? If so, simply list it as a very short constraint, not in terms of HOW that style should be executed.

You will be given the design task to work with below."""
    if is_convergence_branch:
        output += create_convergence_extra_prompt_for_guardrails()
    
    return output



def create_user_constraints_request_prompt(domain_description: str, design_task: str, branch_context: Optional[BranchContext]) -> HumanMessage:
    content = []
    if branch_context is None:
        content.append({"type": "text", "text": f"""
Here's a description of the domain that I'm using:
```
{domain_description}
```

And here's MY design task -- this is what I want the system to help me create/achieve:
```
{design_task}
```

Your role is extremely important, and getting truly creative/novel/interesting outputs is extremely important to me; and so is achieving my design task, and not straying too far from its central spirit.
Go ahead and begin.
"""})
    else:
        if branch_context.reference_img_paths is None or len(branch_context.reference_img_paths) == 0:
            raise ValueError("Branch context must have reference images")
        content.append({"type": "text", "text": f"""
Here's a description of the domain that I'm using:
```
{domain_description}
```

And here's my INITIAL/OVERALL design task -- this is what I want the system to help me create/achieve in general:
```
{design_task}
```

And here is the NEW design task for this branch, consisting of some text I inputted along with some images I selected for additional context:
INPUTTED TEXT:
```
{branch_context.new_design_task}
```
SELECTED IMAGES:
"""})
        for idx, path in enumerate(branch_context.reference_img_paths, start=1):
            data_url = encode_image_to_data_url(path)
            content.append({"type": "text",      "text": f"\n\nSelected Image #{idx}"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        
        content.append({"type": "text", "text": "And here are the IDEAS behind those images:"})
        for idx, idea in enumerate(branch_context.reference_ideas, start=1):
            data_url = encode_image_to_data_url(path)
            content.append({"type": "text",      "text": f"\n\nSelected Image IDEA #{idx}\n```\n{idea}\n```\n\n"})

        content.append({"type": "text", "text": "\n\nYour role is extremely important, and getting truly creative/novel/interesting outputs is extremely important to me; and so is achieving my design task, and not straying too far from its central spirit. Go ahead and begin."})
    return HumanMessage(content=content)