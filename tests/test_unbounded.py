from dataclasses import replace

from evo_core.config import CoreConfig
from evo_core.engine import ResearchKernel
from evo_core.llm import LLMResponse, ToolCallRequest
from evo_core.plugins.subagents import SubagentsPlugin
from evo_core.protocol import BasePlugin, RunState, TurnDecision
from evo_core.system.bootstrap import build_system
from evo_core.system.domain import ExperimentSpec


def test_unbounded_experiment_count_and_duration(tmp_path):
    system = build_system(tmp_path / "system")
    study = system.controller.create("Unlimited experiments", max_experiments=None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text("import json\nfrom pathlib import Path\nPath('outputs').mkdir()\nPath('outputs/metrics.json').write_text(json.dumps({'value':1}))")
    spec = ExperimentSpec("pilot",str(source),("run.py",),"run.py","p","d",timeout_seconds=None)
    for index in range(20):
        system.harness.submit(study.id,1,spec,str(index))
    assert len(system.store.snapshot(study.id)["jobs"]) == 20
    assert system.worker("worker").run_once(study.id).status == "succeeded"


def test_unbounded_coder_exceeds_old_25_calls(tmp_path):
    class Model:
        api_key = None
        calls = 0
        def complete(self, *args, **kwargs):
            self.calls += 1
            raw = {"role":"assistant", "content":"done" if self.calls>30 else ""}
            calls = [] if self.calls>30 else [ToolCallRequest(str(self.calls),"think","{}",{"reflection":"Continue"})]
            return LLMResponse(raw["content"],"",tool_calls=calls,raw_assistant_message=raw)
    cfg = CoreConfig(workspace=str(tmp_path / "ws"),run_dir=str(tmp_path / "meta"))
    cfg.plugins.subagents.max_child_calls = None
    cfg.plugins.subagents.roles = ["coder"]
    plugin = SubagentsPlugin()
    model = Model()
    kernel = ResearchKernel(cfg,plugins=[plugin],llm_client=model)
    ctx = kernel._build_ctx()
    ctx.state = RunState("unbounded","task")
    assert "completed in 31" in plugin.tools(ctx)[0].handler("coder","Continue until finished")


def test_finish_cannot_be_forced_into_unbounded_revisions(tmp_path):
    class Model:
        api_key = None
        def complete(self, *args, **kwargs):
            return LLMResponse("done","",raw_assistant_message={"role":"assistant","content":"done"})
    class Reviewer(BasePlugin):
        name = "reviewer"
        def after_agent_finish(self,state,text,ctx):
            return TurnDecision("revise","Try again","reviewer") if state.revision_count<4 else TurnDecision()
    cfg = CoreConfig(workspace=str(tmp_path / "ws"),run_dir=str(tmp_path / "meta"))
    result = ResearchKernel(cfg,plugins=[Reviewer()],llm_client=Model()).run("task")
    assert result.status == "completed"
    assert result.acceptance == "not_reviewed" and result.model_calls == 1
