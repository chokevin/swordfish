const weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday"];

const roadmap = [
  {
    week: 1,
    focus: "Strict cross-arch GEMM baseline + profiler guardrails",
    doneWhen: "A100/H100/H200 torch/cuBLAS 4096^3 fp16 matrix is complete with passing correctness and NCU; dashboard/report artifacts exist; H200 capacity and A100/DCGM profiler-conflict guardrails are documented; upstream target map and packet generator are ready.",
    readings: "NVIDIA Nsight Compute CLI docs; DCGM exporter/profiling-counter notes; NVIDIA cuBLAS GEMM docs; CUDA C++ Programming Guide performance chapter; Kueue/DRA basics for airun; vLLM contributing guide and recent quant/attention PRs; ONNX Runtime contributing guide; Triton, PyTorch/Inductor, CUTLASS/CuTe, JAX/Pallas, TileLang, and pyptx contribution docs.",
    monday: "Actual: froze M=N=K=4096 fp16 torch/cuBLAS spec, built strict JSON schema/runner, ran complete A100/H100/H200+NCU matrix, fixed H200/A100 blockers, added dashboard/report generation, and created upstream notes/packet tooling.",
    tuesday: "Convert Monday evidence into a clean handoff: 10-line result note, dashboard status check, and first upstream touchpoint candidate chosen from the measured gaps.",
    wednesday: "Run single-GPU Liger per-kernel sweep (RMSNorm/RoPE/SwiGLU/FusedLinearCE) vs HF reference on A100, H100 NVL, and H200; capture latency, peak memory, correctness, and NCU SOL.",
    thursday: "Stretch: reproduce Liger's Llama-3-8B FSDP1 step on 8xA100; extend to 8xH100 NVL / 8xH200 if capacity allows.",
    friday: "Publish cross-arch Liger writeup, generate upstream packet, open a GitHub Discussion on linkedin/Liger-Kernel, and close out the contributions ledger row."
  },
  {
    week: 2,
    focus: "Reproducible runner + first public touchpoint",
    doneWhen: "swordfish has a one-command cluster runner with commit, CUDA, driver, GPU, and NCU metadata; first public upstream issue, PR, benchmark gist, or maintainer-useful discussion comment is live.",
    readings: "Nsight Compute CLI report export docs; Nsight Systems CLI docs; MLPerf Inference reproducibility ideas; vLLM benchmark scripts; upstream issue and PR templates for the chosen first target.",
    monday: "Turn Week 1 JSON into the stable benchmark manifest schema.",
    tuesday: "Write job wrapper for one arch.",
    wednesday: "Generalize wrapper across A100/H100/H200.",
    thursday: "Add NCU/Nsys artifact collection.",
    friday: "Run all three archs from one command, save JSON, and publish the first public touchpoint."
  },
  {
    week: 3,
    focus: "Quant lane statement and internal outreach",
    doneWhen: "FP8/FP4/INT4 lane statement is written; MAI, ORT, and Phi contacts have been messaged.",
    readings: "FP8 Formats for Deep Learning paper; NVIDIA Hopper architecture in-depth blog; NVIDIA Blackwell/FP4 microscaling material; Marlin and DeepGEMM READMEs.",
    monday: "Draft the quant lane statement.",
    tuesday: "List 5 target kernels and 5 target shapes.",
    wednesday: "Identify MAI/ORT/Phi contacts and send short messages.",
    thursday: "Add target repos/files to notes.",
    friday: "Finalize the lane statement and store contact follow-ups."
  },
  {
    week: 4,
    focus: "vLLM, ORT, and Phi profile map",
    doneWhen: "Short notes identify the exact vLLM quant paths, ORT CUDA EP quant paths, and a Phi-4-mini text-only profiling workload worth targeting.",
    readings: "vLLM design docs; vLLM quantization docs; vLLM PagedAttention blog/paper; vLLM Phi-4 recipe; Microsoft Phi-4-mini model card; PhiCookBook profiling/serving notes; ONNX Runtime quantization docs; ORT CUDA EP docs.",
    monday: "Read vLLM engine/scheduler/model-runner overview and Phi-4-mini serving recipe.",
    tuesday: "Trace vLLM quant model load to kernel dispatch.",
    wednesday: "Trace ORT MatMulNBits/quant CUDA paths and confirm the Phi-4-mini support path.",
    thursday: "Define the profile-model contract: TTFT, TPOT, tok/s, prompt/decode shapes, top kernels, and profiler artifact links.",
    friday: "Write top 3 candidate PR opportunities plus the first Phi-4-mini profiling run plan."
  },
  {
    week: 5,
    focus: "Baseline dashboard v0",
    doneWhen: "Public dashboard shows cuBLAS plus one toy kernel across A100/H100/H200.",
    readings: "GitHub Pages docs; simple static dashboard examples; vLLM benchmark output conventions; Nsight Compute metric naming guide.",
    monday: "Choose dashboard data model and public/private boundary.",
    tuesday: "Build minimal static table from Week 1 JSON.",
    wednesday: "Add plots for latency, TFLOPS, SOL.",
    thursday: "Add provenance links: commit/image/command.",
    friday: "Publish dashboard v0 with cuBLAS plus one toy kernel."
  },
  {
    week: 6,
    focus: "First small public contribution",
    doneWhen: "Open one low-risk vLLM or ORT PR: test, doc, benchmark harness, or correctness fix.",
    readings: "vLLM contributing guide; ONNX Runtime contribution guide; both projects test docs; recent small merged PRs in each repo.",
    monday: "Find one low-risk PR target.",
    tuesday: "Reproduce locally and write expected test.",
    wednesday: "Patch the smallest useful change.",
    thursday: "Run targeted tests and write PR description.",
    friday: "Open PR or have a reviewer-ready branch."
  },
  {
    week: 7,
    focus: "A100 matmul learning sprint",
    doneWhen: "Boehm-style matmul reaches a credible A100 baseline with NCU notes.",
    readings: "Simon Boehm CUDA matmul blog; CUDA shared memory chapter; CUDA memory coalescing rules; Nsight Compute memory workload analysis.",
    monday: "Implement naive matmul and baseline against cuBLAS.",
    tuesday: "Add coalesced global loads and shared-memory tiling.",
    wednesday: "Add 2D block tiling and register blocking.",
    thursday: "Profile stalls and memory throughput.",
    friday: "Save A100 NCU notes and next optimization list."
  },
  {
    week: 8,
    focus: "Hopper matmul learning sprint",
    doneWhen: "Hopper wgmma/TMA study kernel runs and produces an NCU writeup.",
    readings: "CUTLASS 3.x GEMM docs; CuTe layout algebra tutorial; Hopper architecture in-depth blog; PTX ISA wgmma/tma sections.",
    monday: "Build CUTLASS Hopper GEMM examples.",
    tuesday: "Read one CuTe GEMM from layout to mainloop.",
    wednesday: "Create minimal WGMMA/TMA study kernel or wrapper.",
    thursday: "Profile pipeline behavior in NCU/Nsys.",
    friday: "Write the Hopper-specific lessons learned."
  },
  {
    week: 9,
    focus: "DeepGEMM reproduction begins",
    doneWhen: "FP8 GEMM reference and test harness are wired; correctness passes for one shape.",
    readings: "DeepGEMM repo/docs; NVIDIA FP8 material; PTX ISA mma/wgmma sections; CUTLASS FP8 examples.",
    monday: "Build DeepGEMM and run reference benchmarks.",
    tuesday: "Create FP8 correctness harness against torch/cuBLAS reference.",
    wednesday: "Wire one shape into swordfish runner.",
    thursday: "Get correctness green on one H100 shape.",
    friday: "Record gap vs DeepGEMM and first bottleneck hypothesis."
  },
  {
    week: 10,
    focus: "DeepGEMM performance pass",
    doneWhen: "FP8 GEMM hits a meaningful fraction of reference performance and bottleneck is known.",
    readings: "Nsight Compute Speed of Light and Roofline docs; CUTLASS profiler docs; DeepGEMM implementation notes/issues.",
    monday: "Run FP8 baseline across target shapes.",
    tuesday: "Identify top stall reason and resource limit.",
    wednesday: "Try one tile/pipeline change.",
    thursday: "Compare against DeepGEMM and cuBLAS/torch where applicable.",
    friday: "Write a short performance note with before/after."
  },
  {
    week: 11,
    focus: "vLLM quant issue with data",
    doneWhen: "File or claim a vLLM quant issue backed by A100/H100/H200 benchmark evidence.",
    readings: "vLLM quantization docs; vLLM benchmark_serving/throughput docs; recent vLLM quant perf issues; PRs touching Marlin/Machete/FP8.",
    monday: "Pick one vLLM quant path with measurable gap.",
    tuesday: "Reproduce vLLM benchmark on A100/H100/H200.",
    wednesday: "Run NCU on the hot kernel.",
    thursday: "Draft issue with command, shape, numbers, and hypothesis.",
    friday: "File/claim the issue and link dashboard evidence."
  },
  {
    week: 12,
    focus: "First real vLLM PR",
    doneWhen: "Merge or actively review a correctness/perf-adjacent vLLM quant PR.",
    readings: "vLLM developer/testing docs; vLLM PR template; recent accepted correctness/perf-adjacent quant PRs.",
    monday: "Branch and reduce scope to mergeable patch.",
    tuesday: "Implement the minimal fix or harness addition.",
    wednesday: "Run unit + targeted benchmark tests.",
    thursday: "Open PR with correctness and numbers.",
    friday: "Respond to first review/CI feedback."
  },
  {
    week: 13,
    focus: "Marlin/Machete reproduction",
    doneWhen: "INT4 weight-only GEMM inner-loop reproduction runs in Triton or CUDA.",
    readings: "Marlin paper and repo; Machete code/docs in vLLM; GPTQ/AWQ weight-only quantization notes; PTX lop3/prmt/dp4a/mma notes as needed.",
    monday: "Read Marlin packing and kernel layout.",
    tuesday: "Implement pack/depack/reference tests.",
    wednesday: "Reproduce INT4 inner loop in Triton or CUDA.",
    thursday: "Benchmark one decode shape vs reference.",
    friday: "Write gap analysis and whether to pursue vLLM PR."
  },
  {
    week: 14,
    focus: "H200 angle v1",
    doneWhen: "Publish a short H100-vs-H200 result where bandwidth changes the bottleneck or tile choice.",
    readings: "H100/H200 product briefs; Hopper memory hierarchy docs; Nsight Compute memory workload/roofline docs; roofline model primer.",
    monday: "Choose a memory-bound kernel candidate.",
    tuesday: "Run H100 vs H200 baseline with identical software.",
    wednesday: "Compare NCU memory throughput and stall reasons.",
    thursday: "Try one H200-specific tile/prefetch change.",
    friday: "Publish short H200 note with numbers and caveats."
  },
  {
    week: 15,
    focus: "ORT candidate implementation",
    doneWhen: "First ORT CUDA EP quant-kernel change is implemented locally with tests.",
    readings: "ONNX Runtime CUDA EP docs; ORT custom/contrib op docs; ORT quantization docs; MatMulNBits operator/kernel source.",
    monday: "Build ORT with CUDA and run targeted tests.",
    tuesday: "Trace target quant kernel from Python/API to CUDA implementation.",
    wednesday: "Implement candidate change locally.",
    thursday: "Run correctness and perf tests.",
    friday: "Prepare PR notes and evidence."
  },
  {
    week: 16,
    focus: "ORT PR",
    doneWhen: "Open or merge the first ORT/ORT GenAI quant PR.",
    readings: "ORT contribution guide; ORT CI/test-selection docs; recent CUDA EP PRs; ORT coding style around contrib ops.",
    monday: "Finalize patch and tests.",
    tuesday: "Run local format/build/test subset.",
    wednesday: "Open ORT or ORT GenAI PR.",
    thursday: "Respond to CI/reviewer comments.",
    friday: "Share PR internally with sponsor/contact."
  },
  {
    week: 17,
    focus: "Dashboard v1",
    doneWhen: "Dashboard includes at least 5 kernels or kernel variants across all three archs.",
    readings: "Benchmark visualization examples; MLPerf result presentation; Nsight Compute metric reference; static site deployment docs.",
    monday: "Add at least 5 kernels/variants to dashboard schema.",
    tuesday: "Backfill A100/H100/H200 runs.",
    wednesday: "Add charts and bottleneck labels.",
    thursday: "Add reproducibility command per row.",
    friday: "Publish dashboard v1 and announce to contacts."
  },
  {
    week: 18,
    focus: "FP4 prototype begins",
    doneWhen: "NVFP4 or MXFP4 prototype is correct against FP16 reference for one shape.",
    readings: "NVIDIA Blackwell/FP4 microscaling docs; OCP MXFP specification material; FP4 quantization papers; CUTLASS low-precision examples.",
    monday: "Choose NVFP4 or MXFP4 scope.",
    tuesday: "Write quant/dequant reference and error tests.",
    wednesday: "Add FP4 GEMM reference path.",
    thursday: "Prototype first kernel or CUTLASS wrapper.",
    friday: "Get one shape correct vs FP16 reference."
  },
  {
    week: 19,
    focus: "FP4 performance pass",
    doneWhen: "FP4 prototype has NCU bottleneck analysis and a clear next optimization.",
    readings: "Nsight Compute roofline docs; CUTLASS low-precision performance notes; relevant FP4/NVFP4 examples and issues.",
    monday: "Benchmark FP4 prototype baseline.",
    tuesday: "Identify bottleneck: memory, tensor core issue, conversion, layout.",
    wednesday: "Try one layout or scaling-data placement change.",
    thursday: "Rerun A100/H100/H200 if supported.",
    friday: "Write optimization note and next PR candidate."
  },
  {
    week: 20,
    focus: "Blog post 1",
    doneWhen: "Publish the cross-arch quant benchmark/matmul learning writeup.",
    readings: "Simon Boehm matmul writeup as style reference; Horace He systems/perf posts; NVIDIA technical blog style examples.",
    monday: "Outline blog post with claims and figures.",
    tuesday: "Generate final dashboard screenshots/plots.",
    wednesday: "Write methodology and caveats.",
    thursday: "Draft and get one technical review.",
    friday: "Publish and post to GPU MODE/internal contacts."
  },
  {
    week: 21,
    focus: "Second vLLM PR",
    doneWhen: "Open or merge a vLLM perf PR with end-to-end decode or throughput numbers.",
    readings: "vLLM quant docs; target kernel source; vLLM e2e benchmark docs; recent perf PR descriptions.",
    monday: "Pick second vLLM PR based on measured bottleneck.",
    tuesday: "Implement patch or autotune/config improvement.",
    wednesday: "Run correctness + microbenchmarks.",
    thursday: "Run e2e decode/throughput benchmark.",
    friday: "Open PR with A100/H100/H200 table."
  },
  {
    week: 22,
    focus: "Internal sponsor loop",
    doneWhen: "One MAI/ORT/Phi sponsor has reviewed the dashboard or a PR and given concrete feedback.",
    readings: "Internal design-doc examples if available; public examples of concise technical RFCs; your dashboard and PR evidence.",
    monday: "Write one-page internal sponsor packet.",
    tuesday: "Ask one MAI/ORT/Phi contact for review.",
    wednesday: "Incorporate feedback and identify their pain point.",
    thursday: "Offer one concrete benchmark they can use.",
    friday: "Record sponsor status and next meeting/action."
  },
  {
    week: 23,
    focus: "MoE/grouped GEMM scout",
    doneWhen: "Decide whether grouped GEMM belongs in scope; write a go/no-go note based on vLLM need.",
    readings: "DeepGEMM grouped GEMM docs; vLLM fused_moe implementation; DeepEP repo/docs; MegaBlocks or related MoE routing paper.",
    monday: "Read vLLM fused_moe path and shapes.",
    tuesday: "Profile one MoE/grouped GEMM workload if available.",
    wednesday: "Compare with DeepGEMM/DeepEP capabilities.",
    thursday: "Write go/no-go criteria for adding MoE to scope.",
    friday: "Decide: include grouped GEMM or defer."
  },
  {
    week: 24,
    focus: "GPU MODE talk proposal",
    doneWhen: "Talk proposal submitted with dashboard and PR evidence.",
    readings: "GPU MODE talk recordings/abstracts; strong CUDA/kernel blog posts; your own dashboard and PR data.",
    monday: "Pick talk title and single thesis.",
    tuesday: "Draft abstract and 5-slide outline.",
    wednesday: "Collect 3 strongest result charts.",
    thursday: "Submit proposal or DM organizer.",
    friday: "Share proposal with sponsor/community contact."
  },
  {
    week: 25,
    focus: "Release-notes PR design",
    doneWhen: "Design doc for the major vLLM PR is posted for maintainer feedback.",
    readings: "vLLM design docs; recent large vLLM quant/kernel PRs; vLLM release notes; maintainer comments on similar proposals.",
    monday: "Choose major PR scope and non-goals.",
    tuesday: "Write design doc with API, tests, and benchmark plan.",
    wednesday: "Post design for maintainer feedback.",
    thursday: "Revise based on feedback.",
    friday: "Lock implementation milestones."
  },
  {
    week: 26,
    focus: "Major vLLM PR implementation",
    doneWhen: "Major quant-kernel PR is functional locally and benchmarked on A100/H100/H200.",
    readings: "Target vLLM kernel/backend code; relevant CUTLASS/Triton docs; prior PR review comments.",
    monday: "Implement core kernel or integration skeleton.",
    tuesday: "Wire feature flag/config path.",
    wednesday: "Add correctness tests.",
    thursday: "Run A100/H100/H200 benchmarks.",
    friday: "Open draft PR or update design with numbers."
  },
  {
    week: 27,
    focus: "Major vLLM PR review",
    doneWhen: "PR is open, reviewers are engaged, and requested changes are being turned around quickly.",
    readings: "vLLM contribution guide; review threads from similar large PRs; project CI failure docs.",
    monday: "Triage all review comments into fix/argue/defer.",
    tuesday: "Address correctness/API comments.",
    wednesday: "Address perf/repro comments.",
    thursday: "Refresh benchmark table.",
    friday: "Post concise reviewer update."
  },
  {
    week: 28,
    focus: "Blog post 2",
    doneWhen: "Publish FP4/FP8 kernel deep dive with NCU screenshots and cross-arch lessons.",
    readings: "NVIDIA FP8/FP4 docs; Marlin/DeepGEMM papers; your NCU reports; strong technical post examples.",
    monday: "Outline FP8/FP4 deep dive.",
    tuesday: "Turn NCU data into figures.",
    wednesday: "Explain failed approaches honestly.",
    thursday: "Get technical review.",
    friday: "Publish and link from dashboard."
  },
  {
    week: 29,
    focus: "ORT second PR",
    doneWhen: "Open or merge a second ORT quant/CUDA EP improvement.",
    readings: "ORT CUDA EP docs; ORT quantization docs; recent ORT CUDA PRs; target source/tests.",
    monday: "Pick second ORT improvement.",
    tuesday: "Implement minimal patch.",
    wednesday: "Run target tests and perf.",
    thursday: "Open PR.",
    friday: "Respond to CI/review and share internally."
  },
  {
    week: 30,
    focus: "Major PR landed or narrowed",
    doneWhen: "Release-notes-worthy vLLM PR is merged, or scope is narrowed to a mergeable subset.",
    readings: "vLLM release process/docs; maintainer feedback on your PR; benchmark dashboard data.",
    monday: "Decide land-now vs narrow-scope.",
    tuesday: "If narrowing, split PR and preserve follow-up issue.",
    wednesday: "Refresh tests/benchmarks.",
    thursday: "Push final revision.",
    friday: "Merge or have an agreed merge path."
  },
  {
    week: 31,
    focus: "Internal pitch",
    doneWhen: "Pitch kernel-benchmark-as-a-service to MAI/ORT/Phi with live dashboard and PR history.",
    readings: "Internal pitch examples; public benchmark dashboard; PR list; one-page product-style memo examples.",
    monday: "Build internal pitch deck/memo.",
    tuesday: "Dry-run with trusted engineer.",
    wednesday: "Pitch MAI/ORT/Phi stakeholder.",
    thursday: "Capture objections and follow-ups.",
    friday: "Convert one team into active consumer/sponsor."
  },
  {
    week: 32,
    focus: "Formal role move",
    doneWhen: "Stretch project, V-team, transfer process, or queued opening is explicitly in motion.",
    readings: "Internal career/transfer docs; brag document examples; staff promotion packet examples if accessible.",
    monday: "Assemble portfolio packet.",
    tuesday: "Talk with current manager about stretch/transfer path.",
    wednesday: "Talk with target manager/sponsor.",
    thursday: "Define written 30/60/90 role plan.",
    friday: "Get explicit next step on calendar."
  },
  {
    week: 33,
    focus: "GPU MODE or internal talk",
    doneWhen: "Talk is delivered or scheduled; slides are backed by real PRs and dashboard data.",
    readings: "GPU MODE talk examples; your blog posts; your dashboard; vLLM/ORT PRs.",
    monday: "Finalize slides.",
    tuesday: "Rehearse technical narrative.",
    wednesday: "Deliver or record talk.",
    thursday: "Publish slides/links.",
    friday: "Follow up with questions, issues, and PR reviewers."
  },
  {
    week: 34,
    focus: "Year-end synthesis",
    doneWhen: "Publish Shipping FP8/FP4 kernels in 2026 with what worked, failed, and shipped.",
    readings: "Your merged PRs; dashboard history; blog posts; NCU notes; vLLM/ORT release notes.",
    monday: "Outline year-end synthesis.",
    tuesday: "Write shipped work and numbers.",
    wednesday: "Write failures and lessons.",
    thursday: "Draft next-year thesis.",
    friday: "Publish and share internally/externally."
  },
  {
    week: 35,
    focus: "Consolidation and next-year ask",
    doneWhen: "Portfolio is packaged for internal move: PR list, dashboard, talks, writeups, sponsor notes.",
    readings: "Your portfolio artifacts; internal feedback; community feedback; next-year roadmap inputs.",
    monday: "Freeze PR/writeup/dashboard list.",
    tuesday: "Prepare one-page portfolio index.",
    wednesday: "Ask sponsors for written feedback or endorsement.",
    thursday: "Draft next-year ask: role/scope/resources.",
    friday: "Send the packet and schedule decision conversations."
  }
];

const storageKey = "swordfish-kernel-roadmap-v1";
let activePhase = "all";
let searchTerm = "";
let checks = loadChecks();

const timeline = document.querySelector("#timeline");
const search = document.querySelector("#search");
const visibleCount = document.querySelector("#visible-count");
const resetProgress = document.querySelector("#reset-progress");
const overallRing = document.querySelector("#overall-ring");
const resultStatus = document.querySelector("#result-status");
const resultTableBody = document.querySelector("#result-table-body");

document.querySelectorAll(".filter").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".filter").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    activePhase = button.dataset.phase;
    render();
  });
});

search.addEventListener("input", (event) => {
  searchTerm = event.target.value.trim().toLowerCase();
  render();
});

resetProgress.addEventListener("click", () => {
  if (window.confirm("Clear local checklist progress for this browser?")) {
    checks = {};
    saveChecks();
    render();
  }
});

render();
loadBenchmarkResults();

function render() {
  const filtered = roadmap.filter((week) => {
    const phaseMatch = activePhase === "all" || getPhase(week.week) === activePhase;
    const haystack = [
      week.focus,
      week.doneWhen,
      week.readings,
      week.monday,
      week.tuesday,
      week.wednesday,
      week.thursday,
      week.friday
    ].join(" ").toLowerCase();
    return phaseMatch && (!searchTerm || haystack.includes(searchTerm));
  });

  timeline.innerHTML = filtered.map(renderWeek).join("");
  visibleCount.textContent = `${filtered.length} of ${roadmap.length} weeks visible`;
  wireCheckboxes();
  updateOverallProgress();
}

function renderWeek(week) {
  const progress = weekProgress(week.week);
  const total = totalWeekChecks(week);
  const complete = progress === total;
  const phase = getPhase(week.week);

  return `
    <article id="week-${week.week}" class="week-card ${complete ? "complete" : ""}" data-phase="${phase}">
      <div class="week-index">
        <div class="week-number">${week.week}</div>
        <span class="phase-pill">${phase}</span>
        <span class="progress-pill">${progress}/${total}</span>
      </div>
      <div class="week-main">
        <h3>${escapeHtml(week.focus)}</h3>
        <p class="done-when">${escapeHtml(week.doneWhen)}</p>
        <div class="reading-box">
          <strong>Noteworthy readings</strong>
          <div class="reading-list">
            ${renderReadingChecks(week)}
          </div>
        </div>
        <div class="day-grid">
          ${weekdays.map((day) => renderDay(week, day)).join("")}
        </div>
      </div>
    </article>
  `;
}

function renderReadingChecks(week) {
  return splitReadings(week.readings).map((reading, index) => {
    const id = readingCheckId(week.week, index);
    const checked = checks[id] ? "checked" : "";
    return `
      <label class="reading-item" for="${id}">
        <input id="${id}" type="checkbox" data-check-id="${id}" ${checked} />
        <span>${escapeHtml(reading)}</span>
      </label>
    `;
  }).join("");
}

function renderDay(week, day) {
  const id = checkId(week.week, day);
  const checked = checks[id] ? "checked" : "";
  return `
    <div class="day-card">
      <label for="${id}">
        <input id="${id}" type="checkbox" data-check-id="${id}" ${checked} />
        <span class="day-name">${day}</span>
        <span class="day-text">${escapeHtml(week[day])}</span>
      </label>
    </div>
  `;
}

function wireCheckboxes() {
  document.querySelectorAll("[data-check-id]").forEach((box) => {
    box.addEventListener("change", (event) => {
      checks[event.target.dataset.checkId] = event.target.checked;
      saveChecks();
      render();
    });
  });
}

function weekProgress(weekNumber) {
  const week = roadmap.find((item) => item.week === weekNumber);
  const dayChecks = weekdays.reduce((total, day) => total + (checks[checkId(weekNumber, day)] ? 1 : 0), 0);
  const readingChecks = splitReadings(week.readings)
    .reduce((total, _reading, index) => total + (checks[readingCheckId(weekNumber, index)] ? 1 : 0), 0);
  return dayChecks + readingChecks;
}

function totalWeekChecks(week) {
  return weekdays.length + splitReadings(week.readings).length;
}

function updateOverallProgress() {
  const totalChecks = roadmap.reduce((sum, week) => sum + totalWeekChecks(week), 0);
  const completed = roadmap.reduce((sum, week) => sum + weekProgress(week.week), 0);
  const percent = Math.round((completed / totalChecks) * 100);
  overallRing.textContent = `${percent}%`;
  overallRing.style.setProperty("--progress", `${percent}%`);
}

function getPhase(weekNumber) {
  if (weekNumber <= 8) return "foundation";
  if (weekNumber <= 19) return "depth";
  if (weekNumber <= 30) return "shipping";
  return "conversion";
}

function checkId(weekNumber, day) {
  return `week-${weekNumber}-${day}`;
}

function readingCheckId(weekNumber, index) {
  return `week-${weekNumber}-reading-${index}`;
}

function splitReadings(readings) {
  return readings
    .split(";")
    .map((reading) => reading.trim().replace(/\.$/, ""))
    .filter(Boolean);
}

function loadChecks() {
  try {
    return JSON.parse(localStorage.getItem(storageKey)) || {};
  } catch {
    return {};
  }
}

function saveChecks() {
  localStorage.setItem(storageKey, JSON.stringify(checks));
}

async function loadBenchmarkResults() {
  if (!resultStatus || !resultTableBody) {
    return;
  }
  try {
    const response = await fetch("results-index.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    renderBenchmarkResults(await response.json());
  } catch (error) {
    resultStatus.textContent = "No result index yet";
    resultTableBody.innerHTML = `
      <tr>
        <td colspan="8">
          Generate one with: uv run python -m swordfish.runner index-results --result-dir /data-nfs/swordfish/week1 --recursive --out docs/dashboard/results-index.json
        </td>
      </tr>
    `;
  }
}

function renderBenchmarkResults(index) {
  const rows = Array.isArray(index.results) ? index.results : [];
  if (rows.length === 0) {
    resultStatus.textContent = "0 result rows";
    resultTableBody.innerHTML = '<tr><td colspan="8">No benchmark results found in the index.</td></tr>';
    return;
  }

  const archs = new Set(rows.map((row) => row.gpu_class).filter(Boolean));
  const missingArchs = ["a100", "h100", "h200"].filter((arch) => !archs.has(arch));
  resultStatus.textContent = `${rows.length} result row${rows.length === 1 ? "" : "s"}${missingArchs.length ? `; missing ${missingArchs.join(", ")}` : "; A100/H100/H200 present"}`;
  resultTableBody.innerHTML = rows
    .slice()
    .sort(compareResultRows)
    .map(renderBenchmarkRow)
    .join("");
}

function compareResultRows(left, right) {
  return [
    String(left.benchmark || "").localeCompare(String(right.benchmark || "")),
    String(left.backend || "").localeCompare(String(right.backend || "")),
    String(left.gpu_class || "").localeCompare(String(right.gpu_class || "")),
    String(left.file || "").localeCompare(String(right.file || ""))
  ].find((value) => value !== 0) || 0;
}

function renderBenchmarkRow(row) {
  const protocolErrors = Array.isArray(row.protocol_errors) ? row.protocol_errors : [];
  const ok = protocolErrors.length === 0 && row.finite_output !== false && row.matches_reference !== false;
  return `
    <tr>
      <td>${escapeHtml(row.file || "")}</td>
      <td>${escapeHtml(row.benchmark || "")}</td>
      <td>${escapeHtml(row.backend || "")}</td>
      <td>${escapeHtml(row.gpu_class || "unknown")}</td>
      <td>${escapeHtml(formatShape(row.shape))}</td>
      <td>${escapeHtml(formatNumber(row.mean_ms))}</td>
      <td>${escapeHtml(formatNumber(row.tflops))}</td>
      <td><span class="status-pill ${ok ? "status-ok" : "status-bad"}">${ok ? "OK" : "Check"}</span></td>
    </tr>
  `;
}

function formatShape(shape) {
  if (!shape || typeof shape !== "object") {
    return "unknown";
  }
  const keys = ["m", "n", "k", "batch_size", "seq_len", "n_embd"];
  const ordered = keys.filter((key) => Object.hasOwn(shape, key));
  Object.keys(shape).forEach((key) => {
    if (!ordered.includes(key)) {
      ordered.push(key);
    }
  });
  return ordered.map((key) => `${key}=${shape[key]}`).join(" ");
}

function formatNumber(value) {
  return typeof value === "number" ? value.toPrecision(4) : "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
