# Workshop Context: Context Engineering for AI-First Development

## The Core Thesis

**Editing code is low leverage. Editing context is high leverage.**

This directory demonstrates context engineering in practice—how investing time in research documents, plan documents, and feedback loops enables AI agents to produce tomorrow's performance with today's models.

## What is Context Engineering?

Context engineering is the practice of designing, structuring, and refining the information environment that AI agents operate within. Instead of perfecting prompts or manually guiding every implementation step, you engineer the context so agents can verify their own work and self-correct.

### High Leverage Activities (What You See Here)

1. **Research documents** - Deep investigation that becomes reusable context
2. **Plan documents** - Structured specifications that agents can execute autonomously
3. **Feedback loops** - Self-verification mechanisms (tests, type checking, linting)
4. **Tool design** - Enabling agents to gather their own context (sub-agents, commands)

### Low Leverage Activities (What You Don't See Here)

1. ❌ Writing perfect prompts for implementation
2. ❌ Manually editing code line-by-line
3. ❌ Explaining the same concept multiple times
4. ❌ Debugging without automated verification

## How This Directory Embodies Context Engineering

### The Context Artifacts

**Research Documents** (`../research/`):
- Deep dives that become permanent context
- Reusable across multiple implementation sessions
- Enable agents to make informed decisions autonomously
- Example: `onnx-cnn-gap-analysis.md` identified the filter-flipping bug before implementation

**Plan Documents** (this directory):
- Executable specifications, not aspirational documents
- Include expected failure modes and verification steps
- Enable test-driven development by AI agents
- Example: `onnx-conv2d-tdd.md` specified every test before implementation

**Inline Annotations** (workshop additions):
- Reality checks showing plan vs execution
- Learning captured for future context
- Makes implicit knowledge explicit

### The Feedback Loops

**Self-Verification Mechanisms Built Into Plans**:
```
Write test → Verify it fails → Implement → Verify it passes
```

Not:
```
Write code → Ask human if it's right → Fix → Ask again
```

**Examples in These Plans**:
- TDD approach in `onnx-conv2d-tdd.md` - agent knows if tests pass/fail
- Coverage requirements - agent can check `pytest --cov`
- Linting checks - agent can run `ruff check`
- Type checking - agent can run `mypy`

The agent doesn't need you to tell it "that's correct"—it verifies itself.

## The Tools That Enable This

### Sub-Agents (Claude Code)
- `codebase-locator` - Agent finds its own reference implementations
- `codebase-analyzer` - Agent deeply understands existing patterns
- `web-search-researcher` - Agent gathers external context when needed

Notice: The plans don't include "step 1: search the codebase for examples." The agent has tools to do that itself.

### Commands (Slash Commands)
- Custom workflows encoded as reusable commands
- Eliminate repetitive explanation
- Build institutional knowledge into tooling

### Test Infrastructure
- `compare_onnx_and_py()` helper - One assertion pattern for all tests
- Hypothesis strategies (future) - Generate test cases automatically
- Structure validation - Verify not just output but internal correctness

## Reading These Plans as Context Engineering Examples

### Example 1: `onnx-backend-implementation.md`
**Context Engineering Lesson**: Initial context was too broad

**What was high leverage**:
- Researching the dispatch pattern once, documenting it
- Identifying the singledispatch architecture that all operations follow
- Creating the comparison test helper that's reused 100+ times

**What was low leverage**:
- Detailed Phase 1-5 breakdown that never got followed exactly
- Trying to spec all edge cases upfront
- Writing code examples that got outdated

**Iteration**: Later plans focused on specific features with tight feedback loops

### Example 2: `onnx-conv2d-tdd.md`
**Context Engineering Lesson**: Perfect context enables autonomous execution

**High leverage context engineering**:
- Asymmetric kernel test specified upfront (catches subtle bugs)
- Expected failure mode documented ("NotImplementedError: No ONNX conversion")
- Verification steps agent can run itself (`pytest test_conv2d_*.py -v`)
- Test data hardcoded in the plan (agent doesn't have to invent it)

**Result**: Agent executed this plan with minimal human intervention because:
1. Tests were specified completely
2. Success criteria were checkable
3. Debugging was guided by test failures
4. No ambiguity about "done"

**The agent's tight feedback loop**:
```
Read plan → Write test → Run test → See NotImplementedError ✓
Implement converter → Run test → See AssertionError (wrong output)
Debug → Fix → Run test → PASSED ✓
Move to next test
```

### Example 3: `hypothesis-property-based-onnx-testing.md`
**Context Engineering Lesson**: Future context, ready when needed

**Why this plan isn't implemented yet**:
- Current feedback loops work (103 tests run fast enough)
- ROI calculation: not worth the learning curve yet
- Context is staged for when it becomes high leverage (150+ tests)

**High leverage aspect**: The plan exists, fully specified, waiting for the right moment. When test count becomes painful, implementation can start immediately. The research cost is already paid.

## Why Research First? Lock Down Scope and Assumptions

Research documents exist to **constrain the agent before implementation starts**.

**Without research context**:
```
Agent: "I'll implement ONNX Conv2D!"
*Starts writing code with wrong assumptions*
*Doesn't know about filter flipping*
*Doesn't know ONNX uses cross-correlation not convolution*
*You spend 3 hours debugging*
```

**With research context**:
```
Agent reads: "ONNX Conv uses cross-correlation, PyTensor allows true convolution via filter_flip=True"
Agent reads: "Must flip kernel when filter_flip=True"
Agent reads: "Use asymmetric kernels to test (symmetric kernels hide the bug)"
*Agent implements correctly first time*
```

**What research locks down**:
- **Scope**: What actually needs to be implemented (not everything, just the essential parts)
- **Assumptions**: ONNX behavior vs PyTensor behavior (prevents misalignment)
- **Critical edge cases**: Filter flipping, padding modes, grouped convolution
- **Mental model**: Architectural patterns (singledispatch, node-based conversion)

**Result**: Agent can't "run wild" with wrong assumptions. The context constrains the solution space.

Example: `onnx-cnn-gap-analysis.md` identified filter-flipping as critical before any code was written. Without this research, the agent would have used symmetric test kernels and never caught the bug.

## Why Plan Next? Lock Down Verification and Feedback

Plans exist to **establish the feedback loop before implementation starts**.

**Without a plan**:
```
Agent: *writes implementation*
You: "Is this right?"
Agent: "I think so?"
You: *manually test*
You: "No, this is wrong"
*Repeat forever*
```

**With a plan**:
```
Plan: "Write test_conv2d_filter_flip_true_asymmetric first"
Plan: "Run it, expect: NotImplementedError: No ONNX conversion available"
Agent: *writes test, runs it* ✓ NotImplementedError (expected!)
Plan: "Implement converter"
Agent: *implements*
Agent: *runs test* ✗ AssertionError (arrays don't match)
Agent: *debugs, finds filter flip bug*
Agent: *fixes*
Agent: *runs test* ✓ PASSED
```

**What plans lock down**:
- **Documentation misalignment**: Tests specify the exact API contract
- **Test design**: Which tests to write, what they verify, what data they use
- **Automated verification**: Commands the agent can run itself (`pytest`, `ruff`, `mypy`)
- **Success criteria**: Checkboxes the agent can verify without asking you
- **Expected failure modes**: Agent knows what "good failure" looks like (NotImplementedError vs crash)

**Result**: Agent has a tight feedback loop. It verifies its own work in seconds, not hours.

Example: `onnx-conv2d-tdd.md` specified exact test code, test data, and expected failures. Agent executed with minimal back-and-forth because every failure was interpretable.

## The Context Problem: Infinite Context Doesn't Exist

If we had infinite context, we'd just dump all research + all plans + entire codebase into every prompt.

**But we don't.**

Context windows are finite. Loading unnecessary information:
- Slows down inference
- Costs money
- Dilutes signal with noise
- Prevents agent from focusing

**The naive approach**:
```
Main context: [research doc] [plan doc] [grep output] [grep output] [ls output]
              [grep output] [file contents] [grep output] [grep output]

Agent: *overwhelmed by 100K tokens of scattered information*
Agent: *can't find the relevant line number*
Agent: "Where was that dispatch pattern again?"
```

**The context-engineered approach**:
```
Main context: [research doc] [plan doc] [current file being edited]

Agent: "I need to find where Elemwise dispatch is implemented"
Agent: *launches codebase-locator sub-agent*
  Sub-agent context: [entire codebase structure] [grep results] [file metadata]
  Sub-agent: "Found it: pytensor/link/onnx/dispatch/elemwise.py:577"
Agent: *receives just the line number*

Main context stays clean. Agent gets exactly what it needs.
```

## Sub-Agents: Context Window Management

Sub-agents are **separate context windows** for separate concerns.

**Instead of polluting your main context with grep/ls/grep/ls**:
```
You: "Find where conv operations are dispatched"
*Main context fills with 50 grep results*
*Main context fills with 20 file listings*
*Main context fills with file contents*
*Agent can barely see the plan anymore*
```

**Use a sub-agent**:
```
You: "Use codebase-locator to find conv dispatch"
*Sub-agent spawns with its own context*
*Sub-agent does grep/ls/grep/read in its context*
*Sub-agent returns: "pytensor/link/onnx/dispatch/conv.py:1276"*
*Main context stays clean*
```

**Sub-agent types and their purpose**:
- **codebase-locator**: "Find where feature X is implemented" → Returns file paths/line numbers
- **codebase-analyzer**: "How does the dispatch pattern work?" → Returns architectural summary
- **web-search-researcher**: "What's ONNX opset 18 Conv behavior?" → Returns external knowledge

**Key insight**: Each sub-agent has its own context window. They do the messy search work, return distilled answers. Your main context stays focused on research + plan + current work.

## Commands: Reusable Context Workflows

Commands encode common workflows so you don't repeat the same context every session.

**Without commands** (low leverage):
```
Session 1: "First read the research, then make a plan with TDD approach, include
           verification steps, document expected failures..."

Session 2: "First read the research, then make a plan with TDD approach, include
           verification steps, document expected failures..."

Session 3: "First read the research, then make a plan with TDD approach..."
```

**With a `/plan-feature` command** (high leverage):
```
.claude/commands/plan-feature.md:
"Read relevant research docs in thoughts/shared/research/
Design a TDD plan with:
- Test specifications (exact code, test data)
- Expected failure modes (NotImplementedError, etc.)
- Verification checklist (pytest, ruff, mypy commands)
- Success criteria (checkboxes agent can verify)
Output to: thoughts/shared/plans/{feature}-tdd.md"

Session 1: /plan-feature onnx-conv2d
Session 2: /plan-feature onnx-pooling
Session 3: /plan-feature onnx-batchnorm
```

**Commands chain with sub-agents**:
```
/implement-plan [plan-file]:
1. Read plan from [plan-file]
2. Use codebase-locator to find reference implementations
3. For each test in plan:
   - Write test
   - Run pytest (verify failure)
   - Implement
   - Run pytest (verify pass)
4. Run verification checklist (ruff, mypy)
5. Report results
```

**Key insight**: Commands are **executable context templates**. Write the workflow once, reuse forever. Combine with sub-agents for powerful automation.

## The Unified Goal: Manage Context, Establish Feedback

Everything we've discussed serves two purposes:

### 1. Manage Context Properly
- **Research docs**: Load domain knowledge once, reference forever
- **Plan docs**: Specify work without cluttering prompt
- **Sub-agents**: Keep search/exploration in separate context windows
- **Commands**: Template common workflows to avoid repetition

**Result**: Main context window contains exactly what's needed for current task, nothing more.

### 2. Establish Tight Feedback Loops
- **Tests**: Agent verifies correctness itself (seconds, not hours)
- **Linting**: Agent checks code quality itself
- **Type checking**: Agent catches errors itself
- **Expected failures**: Agent knows what "good progress" looks like
- **Verification checklists**: Agent knows when it's done

**Result**: Agent self-corrects rapidly without human bottleneck.

## Example: How It All Comes Together

**Task**: Implement ONNX Conv2D support

**Step 1: Research (Lock down assumptions)**
```
Write: thoughts/shared/research/onnx-cnn-gap-analysis.md
Content: ONNX uses cross-correlation, filter flipping is critical, asymmetric kernels required
Context managed: Domain knowledge captured once, reusable
```

**Step 2: Plan (Lock down verification)**
```
Write: thoughts/shared/plans/onnx-conv2d-tdd.md
Content: 21 test specifications, expected failures, verification commands
Context managed: Executable specification, agent can follow autonomously
Feedback loop: Tests provide immediate verification
```

**Step 3: Implement (Agent work with minimal human intervention)**
```
Agent loads: [research] [plan] [current file]
Agent: "Need to find dispatch pattern"
Agent: *launches codebase-locator sub-agent*
  Sub-agent: *searches in its context*
  Returns: "pytensor/link/onnx/dispatch/elemwise.py:577"
Agent: *main context stays clean*
Agent: *writes test per plan*
Agent: *runs pytest* ✗ NotImplementedError (expected!)
Agent: *implements converter*
Agent: *runs pytest* ✗ AssertionError (values wrong)
Agent: *fixes filter flipping*
Agent: *runs pytest* ✓ PASSED
```

**Step 4: Verify (Automated)**
```
Agent: *runs verification checklist from plan*
- pytest tests/link/onnx/test_conv.py -v ✓
- ruff check pytensor/link/onnx/dispatch/conv.py ✓
- mypy pytensor/link/onnx/dispatch/conv.py ✓
Agent: "All verification passed, implementation complete"
```

**Human involvement**:
- Write research (high leverage)
- Write plan (high leverage)
- Review final diff (verification)
- Merge

**Agent autonomy enabled by**:
- Research constraining assumptions
- Plan providing feedback loop
- Sub-agents managing context
- Commands encoding workflows

This is context engineering in practice.

## Key Context Engineering Principles

### 1. Make Verification Automatic
**Don't**: Manually review every line of code
**Do**: Engineer tests that agent can run itself

```markdown
## Success Criteria
- [ ] All tests pass: `pytest tests/link/onnx/test_conv.py -v`
- [ ] Linting passes: `ruff check pytensor/link/onnx/dispatch/conv.py`
```

Agent checks these itself. No human bottleneck.

### 2. Document Failure Modes
**Don't**: "Implement Conv2D support"
**Do**: "Run test, expect NotImplementedError: 'No ONNX conversion available for: AbstractConv2d'"

Agent knows what failure looks like, can recognize when it's on the right track.

### 3. Build Reusable Context
**Don't**: Explain the same architecture every session
**Do**: Write it once in research doc, reference forever

Example: Every ONNX operation references the dispatch pattern research. Never re-explained.

### 4. Optimize the Feedback Loop
**Don't**: Perfect the prompt, then implement everything, then test
**Do**: Write one test, implement until it passes, write next test

Tight loop = agent self-corrects quickly.

### 5. Capture Learning as Context
**Don't**: Let insights evaporate after implementation
**Do**: Annotate plans with reality checks

These annotations become context for the next session.

## The Force Multiplier Effect

### Traditional AI-Assisted Development
```
You: "Implement Conv2D"
AI: *writes code*
You: "That's wrong, fix the filter flipping"
AI: *fixes*
You: "Now the padding is wrong"
AI: *fixes*
You: "Now test it"
...20 rounds of back-and-forth...
```

**Your leverage**: Low (manual QA for everything)

### Context-Engineered AI-First Development
```
You: *Write research doc once*
You: *Write TDD plan once*
AI: *Reads plan, implements 21 tests, runs them, implements converter, all tests pass*
You: *Review diff, merge*
```

**Your leverage**: High (time spent on context scales to many implementations)

## Tomorrow's Performance, Today's Models

These techniques don't require GPT-5 or Claude Opus 4. They work with:
- Claude Sonnet 3.5
- GPT-4
- Any model with function calling and reasonable context window

**The secret**: You're not trying to make the model smarter. You're engineering the context to make success easier.

### Example: The Asymmetric Kernel Insight

The `onnx-conv2d-tdd.md` plan specifies using asymmetric kernels (Sobel filters) to test filter flipping. This isn't the model being smart—this is the human context engineer front-loading domain knowledge.

**Model without this context**: Might use symmetric kernels, miss the bug entirely

**Model with this context**: Uses the exact test that catches the bug, first try

**Result**: Model appears "smarter" because the context is better.

## Using This for Your Workshop

### Workshop Flow

1. **Start with the thesis**: Show the low-leverage vs high-leverage dichotomy
2. **Show a research document**: This is high-leverage context creation
3. **Show a plan document**: This is context that enables autonomy
4. **Show the inline annotations**: This is learning captured as context
5. **Show the actual code**: This is what the agent produced from that context
6. **Discuss the tools**: Sub-agents and commands that gather their own context

### Key Takeaways for Attendees

1. **Invest in context, not prompts** - Research and plans are reusable
2. **Design for self-verification** - Agent shouldn't need you to tell it "correct"
3. **Document failure modes** - Agent learns from expected failures
4. **Build tight feedback loops** - Test-driven, automated checks
5. **Capture learning** - Annotations make implicit knowledge explicit
6. **Use tools that scale** - Sub-agents, commands, test infrastructure

### The Provocative Claim

"You can get 10x more value from today's AI models by investing 10x more time in context engineering and 10x less time in prompt engineering."

These documents prove it: Complex features implemented with minimal back-and-forth because the context was right.

## Questions to Explore in Workshop

1. **What makes good context?** (Complete, verifiable, testable)
2. **When is a plan "good enough"?** (When agent can execute with minimal questions)
3. **How do you know your feedback loop is tight?** (Agent can verify itself in <30 sec)
4. **What belongs in context vs prompts?** (Reusable knowledge vs session-specific instruction)
5. **How do you balance upfront planning vs iterative discovery?** (See the evolution from plan 1 to plan 4)

## Files to Examine

- `onnx-backend-implementation.md` - Context engineering lesson: Too broad, iterated
- `onnx-backend-coverage-and-quality-improvements.md` - Context engineering lesson: Bug-driven, specific
- `onnx-conv2d-tdd.md` - Context engineering lesson: Perfect specificity enables autonomy
- `hypothesis-property-based-onnx-testing.md` - Context engineering lesson: Staged context for future leverage

Each has inline annotations showing plan vs reality, capturing learning as context.

## The Meta Point

**This workshop context document itself is an example of context engineering.**

Instead of explaining these concepts live in the workshop (low leverage), I wrote them once in this document (high leverage). Now:
- You can reference it during the workshop
- Attendees can read it afterward
- Future you can remember the framing
- The concepts are captured permanently

Context engineering, demonstrated recursively.

---

**Remember**: The goal isn't to make AI write perfect code on the first try. The goal is to engineer an environment where AI can verify its own work and self-correct rapidly. That's the leverage multiplier.
