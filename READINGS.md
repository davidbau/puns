# Related Readings

This page collects papers relevant to the study of pun awareness in
language models, organized by topic.

## Cognitive Science: Computational Models of Humor

The computational study of humor is surprisingly underdeveloped.  While
humor has been studied philosophically for centuries — incongruity theory,
superiority theory, and relief theory all date to the Enlightenment —
formal models that can *predict* whether something is funny are rare.
The difficulty is that humor depends on a confluence of factors:
surprise, double meaning, social context, and timing.

The most directly relevant work for our project is
[Kao, Levy, and Goodman (2016)](https://onlinelibrary.wiley.com/doi/10.1111/cogs.12269),
who proposed the first computational model of pun humor grounded in
probabilistic language understanding.  Working at Stanford and UC San Diego,
they formalized two information-theoretic measures — **ambiguity** (how
strongly a sentence supports two readings) and **distinctiveness** (how
different those two readings are) — and showed that both predict human
funniness ratings on puns.  Their key insight is that a good pun maximizes
the product of these two quantities: the word must plausibly mean two very
different things at once.  An earlier version appeared at CogSci 2013.

For a broader survey,
[Loakman, Thorne, and Lin (2025)](https://arxiv.org/abs/2509.21175)
provide a recent overview of computational humor generation and explanation,
noting that despite humor's importance for evaluating language models'
reasoning abilities, research beyond pun generation remains limited and
current systems underperform humans across the board.

## LLMs and Puns

Do large language models actually understand puns, or do they just pattern-match
on surface features?  This question has been addressed directly in recent work,
with results that are both encouraging and cautionary.

[Xu, Yuan, Chen, and Yang (2024)](https://arxiv.org/abs/2404.13599)
("A Good Pun Is Its Own Reword") conducted the first systematic evaluation
of LLMs on pun-related tasks: recognition, explanation, and generation.
Working at Fudan University, they found that while models like GPT-4 can
often identify and explain puns, performance drops sharply on generation tasks
and on puns requiring world knowledge beyond surface-level word similarity.
Their benchmark provides a useful foundation for measuring LLM pun capability.

A more critical perspective comes from
[Zangari, Marcuzzo, Albarelli, Pilehvar, and Camacho-Collados (2025)](https://aclanthology.org/2025.emnlp-main.1419/)
("Pun Unintended"), published at EMNLP 2025.  Their team, spanning the
University of Venice and Cardiff University, showed that while LLMs can detect
puns in standard benchmarks, "their understanding often remains shallow" — models
are easily misled by minor perturbations to wordplay that would not fool humans.
This aligns with our observation that smaller models (3B, 8B) are essentially
pun-blind, and that even 70B+ models require contextual priming to reliably
activate pun completions.

## Superposition, Polysemanticity, and Polysemy

Puns sit at the intersection of two research threads: the
**superposition hypothesis** in mechanistic interpretability, and the study
of **polysemy** in computational linguistics.  Both concern how multiple
meanings coexist in a single representation — superposition in hidden
activations, polysemy in visible vocabulary.

The foundational work on superposition in neural networks is
[Elhage, Hume, Olsson, et al. (2022)](https://transformer-circuits.pub/2022/toy_model/index.html)
("Toy Models of Superposition") from Anthropic.  Using small synthetic
networks, they showed that models learn to represent more features than
they have dimensions by encoding features as nearly-orthogonal directions
in a lower-dimensional space — a form of lossy compression that trades off
interference for capacity.  This was followed by
[Bricken, Templeton, et al. (2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
("Towards Monosemanticity"), which used sparse autoencoders (SAEs) to
decompose superposed representations into interpretable features in a
one-layer transformer.

The theoretical foundations were further developed by
[Scherlis, Sachan, Jermyn, Benton, and Shlegeris (2022)](https://arxiv.org/abs/2210.01892)
("Polysemanticity and Capacity in Neural Networks"), who analyzed how
networks allocate capacity across features of varying importance, finding
that less important features tend toward polysemantic representations.
More recently,
[Foote (2024)](https://arxiv.org/abs/2411.08166)
introduced **neuron embeddings** — a method for identifying distinct
semantic behaviors within individual neurons — to address polysemanticity
in models like GPT-2.

On the linguistic side, the question of how language models handle
word-level polysemy is increasingly well-studied.
[Garí Soler and Apidianaki (2021)](https://arxiv.org/abs/2104.14694)
("Let's Play Mono-Poly") showed that BERT's contextual representations
reflect a word's polysemy level and its partitionability into senses —
meaning that information about how many meanings a word has is encoded
in the model's hidden states.
[Li and Armstrong (2024)](https://onlinelibrary.wiley.com/doi/10.1111/cogs.13416)
extended this to regular polysemy (systematic meaning shifts like
animal → meat or container → contents), finding that BERT embeddings
encode the shared structure within regular polysemy patterns but that
this regularity varies across different relationship types.

For probing how contextualization unfolds within a model,
[Vijayakumar, van Genabith, and Ostermann (2024)](https://arxiv.org/abs/2409.14097)
used linear probes to trace how polysemous words become disambiguated
across different sub-layers of BERT, finding that the process depends
heavily on word position and context window size.  And
[Minegishi, Furuta, Iwasawa, and Matsuo (2025)](https://arxiv.org/abs/2501.06254)
proposed using polysemous words as a lens for evaluating sparse
autoencoders — directly connecting the superposition and polysemy threads
by asking whether SAEs can distinguish different meanings of the same word.

Puns are a uniquely sharp test case at this intersection: they are words
where polysemy is not just present but *foregrounded* — the humor depends
on the listener holding both meanings simultaneously.

## Why Is It Funny?  Neural Substrates of Humor

If pun processing involves simultaneously maintaining two interpretations
and then recognizing their coexistence, what does that look like in the
brain?  Neuroimaging studies of humor provide some clues.

[Dai, Chen, Chan, Wu, Li, Cho, and Hu (2017)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2017.00498/full)
proposed a **dual-path model of incongruity resolution** using fMRI.
Working at National Taiwan Normal University, they distinguished two types
of humor: jokes with resolvable incongruities (where the punchline makes
sense once you see both meanings) and absurd humor (where the incongruity
is unresolvable).  They found distinct neural pathways for each — the
resolvable pathway engages integration areas in the prefrontal cortex,
while absurd humor engages different regions.  Puns are firmly in the
"resolvable" category: the humor comes precisely from resolving the
double meaning.

Earlier,
[Samson, Hempelmann, Huber, and Zysset (2009)](https://pubmed.ncbi.nlm.nih.gov/19046978/)
used fMRI to compare incongruity-resolution humor (like puns) with
nonsense humor, finding that incongruity-resolution cartoons selectively
engage prefrontal regions associated with coherence-building and semantic
integration.  This is consistent with the idea that pun processing requires
actively constructing a bridge between two meanings — a process that may
share neural substrates with general-purpose ambiguity resolution.

These findings motivate a key question in our project: if we find internal
features in language models that drive pun awareness, do those features
also serve a broader role in semantic integration and ambiguity resolution?
Or is pun processing a narrow, isolated capability?
