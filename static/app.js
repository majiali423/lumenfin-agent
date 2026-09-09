/* LumenFin demo UI: real job polling, no simulated node timers. */
(function () {
  "use strict";

  var JOB_STORAGE_KEY = "lumenfin.active_job_id";
  var TERMINAL = { completed: 1, failed: 1 };
  var chartInstances = {};
  var uploadedFiles = [];
  var lastResult = null;
  var pendingClarification = null;
  var rawView = "manifest";
  var pollTimer = null;
  var activeJobId = null;
  var benchReturnFocus = null;

  function $(id) {
    return document.getElementById(id);
  }

  function refreshIcons() {
    if (window.lucide && typeof window.lucide.createIcons === "function") {
      window.lucide.createIcons();
    }
  }

  function showToast(msg, type) {
    type = type || "error";
    var el = document.createElement("div");
    el.className = "toast toast-" + type;
    el.textContent = msg;
    $("toastContainer").appendChild(el);
    setTimeout(function () {
      el.remove();
    }, 5000);
  }

  function apiHeaders(base) {
    var headers = Object.assign({}, base || {});
    var input = $("apiKey");
    var key = input ? input.value.trim() : "";
    if (key) headers["X-API-Key"] = key;
    return headers;
  }

  function apiFetch(url, options) {
    var request = Object.assign({}, options || {});
    request.headers = apiHeaders(request.headers);
    return fetch(url, request);
  }

  async function apiError(r) {
    var err = await r.json().catch(function () {
      return {};
    });
    if (r.status === 401) {
      return new Error("请在高级设置中输入与服务端 MAS_API_KEY 一致的 API Key。");
    }
    return new Error(err.detail || r.statusText);
  }

  function markdownToHtml(md) {
    var parseFn =
      typeof marked !== "undefined" && marked.parse
        ? function (text) {
            return marked.parse(text);
          }
        : null;
    return LumenFinSanitize.renderMarkdown(md, parseFn);
  }

  function selectedOutputFormat() {
    var el = document.querySelector('input[name="outputFormat"]:checked');
    return el ? el.value : "research_report";
  }

  function useExampleQuery(kind) {
    var examples = {
      NVIDIA:
        "结合 NVIDIA FY2025 财报，分析数据中心业务增长、毛利率变化与主要投资风险，并给出可核验引用。",
      Apple: "分析 Apple FY2024 的盈利能力、自由现金流质量与主要风险，区分已验证结论和数据缺口。",
      Compare: "比较 Apple 与 Microsoft 最近财年的收入增长、利润率和现金流质量，并说明证据来源。",
    };
    $("query").value = examples[kind] || "";
    $("query").focus();
  }

  function setCompanyScope(scope) {
    $("clarifyCompanyScope").value = scope;
    $("clarifyScopeHint").textContent = "已选 company_scope=" + scope;
  }

  function rememberJob(jobId) {
    activeJobId = jobId || null;
    if (jobId) {
      try {
        sessionStorage.setItem(JOB_STORAGE_KEY, jobId);
      } catch (e) {}
      if (window.history && history.replaceState) {
        history.replaceState(null, "", "?job=" + encodeURIComponent(jobId));
      }
    } else {
      try {
        sessionStorage.removeItem(JOB_STORAGE_KEY);
      } catch (e) {}
      if (window.history && history.replaceState) {
        history.replaceState(null, "", window.location.pathname);
      }
    }
    var badge = $("jobIdBadge");
    if (badge) {
      LumenFinSanitize.setText(badge, jobId ? "job " + jobId : "");
    }
  }

  function jobIdFromLocation() {
    var params = new URLSearchParams(window.location.search || "");
    return (params.get("job") || "").trim();
  }

  function resetProgress() {
    var host = $("progressSteps");
    if (host) host.textContent = "";
    var title = $("progressTitle");
    if (title) title.textContent = "作业运行中（等待真实节点事件）";
  }

  function applyAuditProgress(auditLog) {
    var host = $("progressSteps");
    if (!host) return;
    var events = auditLog || [];
    if (!events.length) {
      resetProgress();
      return;
    }
    $("progressTitle").textContent = "真实执行轨迹";
    host.textContent = "";
    events.forEach(function (event, index) {
      var step = document.createElement("div");
      var status = String(event.status || "");
      step.className = "progress-step " + (status === "ok" ? "done" : "active");
      step.setAttribute("data-step", String(event.step || ""));
      var ind = document.createElement("div");
      ind.className = "progress-indicator " + (status === "ok" ? "done" : "active");
      ind.textContent = String(index + 1);
      var label = document.createElement("div");
      LumenFinSanitize.setText(label, event.step || "node");
      step.appendChild(ind);
      step.appendChild(label);
      host.appendChild(step);
    });
  }

  function setBusy(isBusy) {
    var btn = $("submitBtn");
    var btnText = $("btnText");
    var spinner = $("btnSpinner");
    btn.disabled = isBusy;
    btnText.textContent = isBusy ? "正在分析" : "开始分析";
    spinner.style.display = isBusy ? "inline-block" : "none";
  }

  function showRunning() {
    $("emptyState").style.display = "none";
    $("resultsArea").style.display = "none";
    $("clarifyState").style.display = "none";
    $("failState").style.display = "none";
    $("progressPanel").style.display = "";
    $("loadingState").style.display = "";
  }

  function hideRunning() {
    $("loadingState").style.display = "none";
  }

  function destroyCharts() {
    Object.keys(chartInstances).forEach(function (key) {
      try {
        chartInstances[key].destroy();
      } catch (e) {}
    });
    chartInstances = {};
  }

  async function checkHealth() {
    try {
      var responses = await Promise.all([fetch("/health"), fetch("/ready")]);
      var r = responses[0];
      var readyResponse = responses[1];
      if (!r.ok) throw new Error("health check failed");
      var d = await r.json();
      var ready = await readyResponse.json().catch(function () {
        return { checks: {} };
      });
      $("statusDot").className = "status-dot online";
      var marketOk = d.market_provider_ok;
      var marketLabel = d.market_provider || "yahoo";
      $("statusLabel").textContent = "在线 · " + marketLabel + (marketOk ? " 可用" : " 异常");
      var rag = $("ragBadge");
      var milvus = (ready.checks || {}).milvus;
      if (!d.rag_enabled) {
        rag.textContent = "RAG 未启用";
        rag.className = "nav-badge warning";
      } else if (milvus && milvus.ok) {
        rag.textContent = "RAG 可用";
        rag.className = "nav-badge";
      } else {
        rag.textContent = "RAG 异常";
        rag.className = "nav-badge offline";
      }
      var b = $("llmBadge");
      b.textContent = d.llm_backend === "deepseek" ? "DeepSeek" : "Fallback";
      b.className = d.llm_backend === "deepseek" ? "nav-badge" : "nav-badge offline";
    } catch (e) {
      $("statusDot").className = "status-dot offline";
      $("statusLabel").textContent = "服务未连接";
      $("ragBadge").textContent = "RAG 未知";
      $("ragBadge").className = "nav-badge offline";
    }
  }

  async function loadConfig() {
    try {
      var r = await apiFetch("/api/v1/config");
      if (!r.ok) return;
      var d = await r.json();
      $("configInfo").textContent =
        "模型：" + d.deepseek_model + " · 市场数据：" + d.market_data_provider + " · 数据模式：" + (d.data_mode || "—");
      var mode = $("dataModeBadge");
      if (mode) {
        var label = d.data_mode === "live" ? "LIVE" : "DEMO";
        LumenFinSanitize.setText(mode, label);
        mode.className = "nav-badge " + (d.data_mode === "live" ? "warning" : "");
        mode.title = "本次会话 DATA_MODE=" + (d.data_mode || "unknown") + "；与历史评测分数不是同一口径";
      }
    } catch (e) {}
  }

  function isSupportedUploadFile(file) {
    var name = ((file && file.name) || "").toLowerCase();
    return [".pdf", ".csv", ".xlsx", ".md", ".markdown", ".json"].some(function (ext) {
      return name.endsWith(ext);
    });
  }

  function addUploadFiles(fileList) {
    var added = 0;
    var skipped = 0;
    for (var i = 0; i < fileList.length; i++) {
      if (isSupportedUploadFile(fileList[i])) {
        uploadedFiles.push(fileList[i]);
        added++;
      } else skipped++;
    }
    if (skipped) showToast("已跳过 " + skipped + " 个不支持的文件（仅 PDF/CSV/XLSX/MD/JSON）", "error");
    if (added) updateFileList();
  }

  function updateFileList() {
    var names = uploadedFiles.map(function (f) {
      return f.name;
    }).join(", ");
    $("fileList").textContent = names;
    var uz = $("uploadZone");
    if (uploadedFiles.length) {
      uz.classList.add("has-files");
      $("uploadText").textContent = uploadedFiles.length + " 个文件已选择";
      $("uploadHint").textContent = "可继续拖拽文件，或点击添加";
    } else {
      uz.classList.remove("has-files");
      $("uploadText").textContent = "拖拽文件到这里，或点击选择";
      $("uploadHint").textContent = "支持 PDF、CSV、XLSX、Markdown、JSON";
    }
  }

  function bindUpload() {
    var fi = $("fileInput");
    var uz = $("uploadZone");
    var uploadDragDepth = 0;
    fi.addEventListener("change", function (e) {
      addUploadFiles(e.target.files);
      e.target.value = "";
    });
    uz.addEventListener("dragenter", function (e) {
      e.preventDefault();
      uploadDragDepth++;
      uz.classList.add("is-dragging");
      $("uploadText").textContent = "松开即可添加文件";
    });
    uz.addEventListener("dragover", function (e) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    });
    uz.addEventListener("dragleave", function (e) {
      e.preventDefault();
      uploadDragDepth = Math.max(0, uploadDragDepth - 1);
      if (uploadDragDepth === 0) {
        uz.classList.remove("is-dragging");
        updateFileList();
      }
    });
    uz.addEventListener("drop", function (e) {
      e.preventDefault();
      uploadDragDepth = 0;
      uz.classList.remove("is-dragging");
      addUploadFiles(e.dataTransfer.files);
      updateFileList();
    });
    uz.addEventListener("click", function () {
      fi.click();
    });
    uz.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fi.click();
      }
    });
    updateFileList();
  }

  function jobToAnalyzePayload(job) {
    var result = job.result || {};
    return {
      thread_id: job.thread_id || result.thread_id,
      job_id: job.job_id,
      llm_backend: job.llm_backend || result.llm_backend,
      workflow_status: result.workflow_status || job.workflow_status || job.status,
      clarification_questions: result.clarification_questions || [],
      final_report: result.final_report || "",
      executive_summary: result.executive_summary,
      answer: result.answer,
      citations: result.citations || [],
      audit_log: job.audit_log || result.audit_log || [],
      artifacts: job.artifacts || {},
      state: result,
      chart_data: result.chart_data,
      run_telemetry: result.run_telemetry,
      run_manifest: result.run_manifest,
      error_message: job.error_message,
    };
  }

  function showFail(message) {
    hideRunning();
    $("clarifyState").style.display = "none";
    $("resultsArea").style.display = "none";
    $("failState").style.display = "";
    $("progressPanel").style.display = "";
    LumenFinSanitize.setText($("failMessage"), message || "分析失败");
  }

  function showClarificationPanel(d) {
    pendingClarification = d;
    hideRunning();
    $("failState").style.display = "none";
    $("clarifyState").style.display = "";
    var qs = d.clarification_questions || [];
    var list = $("clarifyQuestions");
    list.textContent = "";
    (qs.length ? qs : ["请补充公司与分析时间范围。"]).forEach(function (q) {
      var li = document.createElement("li");
      LumenFinSanitize.setText(li, q);
      list.appendChild(li);
    });
    $("clarifyCompany").value = "";
    $("clarifyTimeRange").value = "";
    $("clarifyCompanyScope").value = "";
    $("clarifyScopeHint").textContent = "未选择 scope";
    refreshIcons();
  }

  function switchTab(name) {
    document.querySelectorAll(".tab-btn").forEach(function (t) {
      t.classList.remove("active");
    });
    document.querySelectorAll(".tab-panel").forEach(function (c) {
      c.classList.remove("active");
    });
    var tab = document.querySelector('.tab-btn[data-tab="' + name + '"]');
    if (tab) tab.classList.add("active");
    var panel = $("tab-" + name);
    if (panel) panel.classList.add("active");
    if (name === "charts" && lastResult) setTimeout(function () {
      renderCharts(lastResult);
    }, 80);
  }

  function switchRawView(view) {
    rawView = view;
    document.querySelectorAll(".raw-subtab").forEach(function (btn) {
      btn.classList.remove("active");
    });
    var active = document.querySelector('.raw-subtab[data-raw="' + view + '"]');
    if (active) active.classList.add("active");
    if (!lastResult) {
      $("stateContent").textContent = "{}";
      return;
    }
    var payload = lastResult;
    if (view === "manifest") payload = lastResult.run_manifest || {};
    else if (view === "state") payload = lastResult.state || {};
    $("stateContent").textContent = JSON.stringify(payload, null, 2);
  }

  function renderEvidence(d) {
    var host = $("evidenceList");
    host.textContent = "";
    var citations = d.citations || [];
    var note = $("evidenceNote");
    if (!citations.length) {
      LumenFinSanitize.setText(note, "本次没有已验证 chunk 引用。不可验证的自由文本不算已引用。");
      return;
    }
    LumenFinSanitize.setText(note, "点击引用可对照简洁回答中的 [n] 标记。这些是本 run 的稳定 chunk id，不是历史评测分数。");
    citations.forEach(function (cid, i) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "query-example";
      LumenFinSanitize.setText(btn, "[" + (i + 1) + "] " + cid);
      btn.addEventListener("click", function () {
        switchTab("answer");
        showToast("证据 " + cid, "success");
      });
      host.appendChild(btn);
    });
  }

  function renderFormulas(d) {
    var host = $("formulaTable");
    host.textContent = "";
    var metrics = (d.state || {}).financial_metrics || {};
    var companies = Object.keys(metrics);
    if (!companies.length) {
      LumenFinSanitize.setText(host, "本次没有 AST 可计算比率输入。");
      return;
    }
    var table = document.createElement("table");
    var thead = document.createElement("thead");
    var headRow = document.createElement("tr");
    ["公司", "指标", "值"].forEach(function (label) {
      var th = document.createElement("th");
      LumenFinSanitize.setText(th, label);
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    var body = document.createElement("tbody");
    table.appendChild(thead);
    table.appendChild(body);
    companies.forEach(function (company) {
      var rowMetrics = metrics[company] || {};
      Object.keys(rowMetrics).forEach(function (key) {
        var tr = document.createElement("tr");
        var td1 = document.createElement("td");
        var td2 = document.createElement("td");
        var td3 = document.createElement("td");
        LumenFinSanitize.setText(td1, company);
        LumenFinSanitize.setText(td2, key);
        LumenFinSanitize.setText(td3, String(rowMetrics[key]));
        tr.appendChild(td1);
        tr.appendChild(td2);
        tr.appendChild(td3);
        body.appendChild(tr);
      });
    });
    host.appendChild(table);
  }

  function renderManifestPanel(d) {
    var m = d.run_manifest || {};
    var tel = d.run_telemetry || (d.state || {}).run_telemetry || {};
    var latency = m.total_latency_ms != null ? m.total_latency_ms : tel.total_latency_ms || 0;
    var promptTok = m.total_prompt_tokens != null ? m.total_prompt_tokens : tel.total_prompt_tokens || 0;
    var completionTok =
      m.total_completion_tokens != null ? m.total_completion_tokens : tel.total_completion_tokens || 0;
    var score = m.evaluator_score != null ? m.evaluator_score : "—";
    var grade = m.evaluator_grade || "";
    var metrics = [
      { k: "Workflow", v: m.workflow_status || d.workflow_status || "—" },
      { k: "Job", v: d.job_id || activeJobId || "—" },
      { k: "Latency", v: latency + " ms" },
      { k: "Tokens In/Out", v: promptTok + " / " + completionTok },
      { k: "This-run evaluator", v: score + (grade ? " (" + grade + ")" : "") },
      { k: "Data mode", v: (d.state || {}).data_mode || "—" },
    ];
    var metricsHtml =
      '<div class="manifest-section-title">本次运行（不是历史合同分）</div><div class="manifest-grid">' +
      metrics
        .map(function (x) {
          return (
            '<div class="manifest-item"><div class="manifest-key">' +
            LumenFinSanitize.escapeText(x.k) +
            '</div><div class="manifest-val">' +
            LumenFinSanitize.escapeText(String(x.v)) +
            "</div></div>"
          );
        })
        .join("") +
      "</div>";
    var arts = d.artifacts || {};
    var artKeys = Object.keys(arts);
    var artifactsHtml = artKeys.length
      ? '<div class="manifest-section-title">导出</div><ul>' +
        artKeys
          .map(function (k) {
            return "<li>" + LumenFinSanitize.escapeText(k + ": " + arts[k]) + "</li>";
          })
          .join("") +
        "</ul>"
      : "";
    LumenFinSanitize.setSanitizedHtml($("manifestPanel"), metricsHtml + artifactsHtml);
  }

  function renderCharts(d) {
    destroyCharts();
    if (typeof Chart === "undefined") {
      var empty = $("chartUnavailable");
      if (empty) empty.style.display = "";
      return;
    }
    var empty = $("chartUnavailable");
    if (empty) empty.style.display = "none";
    var cd = d.chart_data || (d.state || {}).chart_data || {};
    var tc = function (id, has) {
      var el = $(id);
      if (el) {
        el.style.display = has ? "" : "none";
      }
    };
    var mc = cd.metrics_comparison;
    if (mc && mc.datasets && mc.datasets.length > 0) {
      var ctxM = $("chartMetrics");
      if (ctxM)
        try {
          chartInstances.metrics = new Chart(ctxM.getContext("2d"), {
            type: "bar",
            data: mc,
            options: { responsive: true, maintainAspectRatio: false },
          });
          tc("chartMetrics", true);
        } catch (e) {
          tc("chartMetrics", false);
        }
    }
    var tl = cd.agent_timeline || d.audit_log || [];
    $("agentTimeline").textContent = "";
    if (!tl.length) {
      LumenFinSanitize.setText($("agentTimeline"), "暂无时间线");
      return;
    }
    tl.forEach(function (e, i) {
      var row = document.createElement("div");
      row.style.cssText = "display:flex;gap:10px;padding:7px 0;border-bottom:1px solid var(--border-subtle);";
      var n = document.createElement("span");
      n.style.cssText = "font-size:10px;color:#94a3b8;width:20px;font-family:var(--font-mono);";
      n.textContent = String(i + 1).padStart(2, "0");
      var s = document.createElement("span");
      LumenFinSanitize.setText(s, e.step || "");
      var st = document.createElement("span");
      LumenFinSanitize.setText(st, e.status || "");
      row.appendChild(n);
      row.appendChild(s);
      row.appendChild(st);
      $("agentTimeline").appendChild(row);
    });
  }

  function renderResults(d) {
    lastResult = d;
    $("clarifyState").style.display = "none";
    $("failState").style.display = "none";
    $("resultsArea").style.display = "";
    hideRunning();
    $("resultBackend").textContent = "Backend: " + String(d.llm_backend || "unknown").toUpperCase();
    $("resultThread").textContent = "Session: " + (d.thread_id || "-");
    var status = d.workflow_status || "";
    var banner = $("statusBanner");
    banner.style.display = "";
    if (status === "incomplete_data") {
      LumenFinSanitize.setText(banner, "资料不足：已返回局部结论，未发明数值。");
      banner.className = "status-banner warn";
    } else if (status === "blocked_by_guardrail") {
      LumenFinSanitize.setText(banner, "输入被护栏阻断。");
      banner.className = "status-banner fail";
    } else {
      LumenFinSanitize.setText(banner, "工作流状态：" + status);
      banner.className = "status-banner";
    }
    var concise = d.answer || d.executive_summary || "";
    LumenFinSanitize.setSanitizedHtml($("answerContent"), markdownToHtml(concise || "（无简洁回答字段，见完整报告）"));
    LumenFinSanitize.setSanitizedHtml($("reportContent"), markdownToHtml(d.final_report || ""));
    renderEvidence(d);
    renderFormulas(d);
    applyAuditProgress(d.audit_log || (d.state || {}).audit_log || []);
    renderManifestPanel(d);
    switchRawView("manifest");
    var auditLog = d.audit_log || (d.state || {}).audit_log || [];
    var auditHost = $("auditTimeline");
    auditHost.textContent = "";
    if (!auditLog.length) {
      LumenFinSanitize.setText(auditHost, "没有真实节点事件（不会用计时器假装走过各阶段）。");
    } else {
      auditLog.forEach(function (e) {
        var item = document.createElement("div");
        item.className = "audit-item";
        var title = document.createElement("div");
        title.className = "audit-title";
        LumenFinSanitize.setText(title, e.step + " · " + (e.status || ""));
        var desc = document.createElement("div");
        desc.className = "audit-desc";
        LumenFinSanitize.setText(desc, e.detail || "");
        item.appendChild(title);
        item.appendChild(desc);
        auditHost.appendChild(item);
      });
    }
    $("auditContent").textContent = JSON.stringify(auditLog, null, 2);
    switchTab("answer");
    refreshIcons();
  }

  function handleTerminalJob(job) {
    if (job.status === "failed") {
      showFail(job.error_message || "后台作业失败");
      setBusy(false);
      return;
    }
    var payload = jobToAnalyzePayload(job);
    if (payload.workflow_status === "needs_clarification") {
      showClarificationPanel(payload);
      showToast("需要补充公司与时间范围信息", "success");
      setBusy(false);
      return;
    }
    renderResults(payload);
    showToast("分析已完成", "success");
    setBusy(false);
  }

  async function pollJob(jobId) {
    var r = await apiFetch("/api/v1/jobs/" + encodeURIComponent(jobId));
    if (!r.ok) throw await apiError(r);
    var job = await r.json();
    applyAuditProgress(job.audit_log || (job.result || {}).audit_log || []);
    if (!TERMINAL[job.status]) {
      var title = $("progressTitle");
      if (title) title.textContent = "作业 " + job.status + "（仅显示已发生的节点）";
      return job;
    }
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    handleTerminalJob(job);
    return job;
  }

  function startPolling(jobId) {
    rememberJob(jobId);
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(function () {
      pollJob(jobId).catch(function (e) {
        if (pollTimer) {
          clearInterval(pollTimer);
          pollTimer = null;
        }
        showFail(e.message);
        setBusy(false);
      });
    }, 700);
    return pollJob(jobId);
  }

  async function runAnalysis() {
    var query = $("query").value.trim();
    if (!query) {
      showToast("请先输入研究问题");
      return;
    }
    var outputFormat = selectedOutputFormat();
    setBusy(true);
    resetProgress();
    showRunning();
    destroyCharts();
    try {
      var r;
      if (uploadedFiles.length > 0) {
        var fd = new FormData();
        fd.append("query", query);
        fd.append("thread_id", $("threadId").value || "");
        fd.append("export_artifacts", "true");
        fd.append("output_format", outputFormat);
        for (var i = 0; i < uploadedFiles.length; i++) fd.append("files", uploadedFiles[i]);
        r = await apiFetch("/api/v1/jobs/upload", { method: "POST", body: fd });
      } else {
        r = await apiFetch("/api/v1/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: query,
            thread_id: $("threadId").value || undefined,
            export_artifacts: true,
            output_format: outputFormat,
          }),
        });
      }
      if (!r.ok) throw await apiError(r);
      var created = await r.json();
      if (created.thread_id) $("threadId").value = created.thread_id;
      await startPolling(created.job_id);
    } catch (e) {
      showFail(e.message);
      setBusy(false);
    }
  }

  async function submitClarification() {
    if (!pendingClarification || !pendingClarification.thread_id) {
      showToast("No pending clarification session");
      return;
    }
    var company = $("clarifyCompany").value.trim();
    var timeRange = $("clarifyTimeRange").value.trim();
    var companyScope = $("clarifyCompanyScope").value.trim();
    if (!company && !timeRange && !companyScope) {
      showToast("请填写公司/时间范围，或选择 company_scope");
      return;
    }
    var btn = $("clarifyBtn");
    var btnText = $("clarifyBtnText");
    var spinner = $("clarifySpinner");
    btn.disabled = true;
    btnText.textContent = "正在继续";
    spinner.style.display = "inline-block";
    $("clarifyState").style.display = "none";
    showRunning();
    try {
      var clarification = {};
      if (company) clarification.company = company;
      if (timeRange) clarification.time_range = timeRange;
      if (companyScope) clarification.company_scope = companyScope;
      var r = await apiFetch("/api/v1/clarify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          thread_id: pendingClarification.thread_id,
          clarification: clarification,
          export_artifacts: true,
          job_id: activeJobId || pendingClarification.job_id || undefined,
        }),
      });
      if (!r.ok) throw await apiError(r);
      var d = await r.json();
      d.job_id = activeJobId || pendingClarification.job_id || d.job_id;
      if (d.workflow_status === "needs_clarification") {
        showClarificationPanel(d);
        showToast("仍需补充信息", "success");
      } else {
        pendingClarification = null;
        renderResults(d);
        showToast("已从同一 checkpoint 继续", "success");
      }
    } catch (e) {
      hideRunning();
      $("clarifyState").style.display = "";
      showToast("Resume failed: " + e.message);
    } finally {
      btn.disabled = false;
      btnText.textContent = "补充并继续";
      spinner.style.display = "none";
    }
  }

  function clearResults() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    rememberJob(null);
    $("resultsArea").style.display = "none";
    $("clarifyState").style.display = "none";
    $("failState").style.display = "none";
    $("progressPanel").style.display = "none";
    $("emptyState").style.display = "";
    destroyCharts();
    lastResult = null;
    pendingClarification = null;
    uploadedFiles = [];
    updateFileList();
    resetProgress();
  }

  async function restoreJobIfAny() {
    var jobId = jobIdFromLocation();
    if (!jobId) {
      try {
        jobId = sessionStorage.getItem(JOB_STORAGE_KEY) || "";
      } catch (e) {
        jobId = "";
      }
    }
    if (!jobId) return;
    showRunning();
    setBusy(true);
    try {
      await startPolling(jobId);
    } catch (e) {
      showFail("无法恢复作业： " + e.message);
      setBusy(false);
    }
  }

  function openBenchDrawer() {
    var overlay = $("benchOverlay");
    var trigger = $("benchTrigger");
    benchReturnFocus = document.activeElement;
    overlay.classList.add("open");
    overlay.setAttribute("aria-hidden", "false");
    trigger.setAttribute("aria-expanded", "true");
    document.body.classList.add("bench-open");
    window.setTimeout(function () {
      $("benchClose").focus();
    }, 80);
  }

  function closeBenchDrawer() {
    var overlay = $("benchOverlay");
    var trigger = $("benchTrigger");
    overlay.classList.remove("open");
    overlay.setAttribute("aria-hidden", "true");
    trigger.setAttribute("aria-expanded", "false");
    document.body.classList.remove("bench-open");
    if (benchReturnFocus && typeof benchReturnFocus.focus === "function") benchReturnFocus.focus();
  }

  window.refreshIcons = refreshIcons;
  window.useExampleQuery = useExampleQuery;
  window.setCompanyScope = setCompanyScope;
  window.runAnalysis = runAnalysis;
  window.submitClarification = submitClarification;
  window.clearResults = clearResults;
  window.switchTab = switchTab;
  window.switchRawView = switchRawView;
  window.openBenchDrawer = openBenchDrawer;
  window.closeBenchDrawer = closeBenchDrawer;
  window.pollJob = pollJob;
  window.restoreJobIfAny = restoreJobIfAny;

  document.addEventListener("DOMContentLoaded", function () {
    bindUpload();
    refreshIcons();
    checkHealth();
    loadConfig();
    setInterval(checkHealth, 30000);
    $("apiKey").addEventListener("change", loadConfig);
    $("query").addEventListener("keydown", function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        runAnalysis();
      }
    });
    $("benchOverlay").addEventListener("click", function (event) {
      if (event.target === event.currentTarget) closeBenchDrawer();
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && $("benchOverlay").classList.contains("open")) closeBenchDrawer();
    });
    restoreJobIfAny();
  });
})();
