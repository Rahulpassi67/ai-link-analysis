
const API_URL = "/link-analysis";
const form = document.getElementById("analysisForm");
const urlInput = document.getElementById("websiteUrl");
const analyzeBtn = document.getElementById("analyzeBtn");
const buttonText = document.getElementById("buttonText");
const loadingPanel = document.getElementById("loadingPanel");
const errorPanel = document.getElementById("errorPanel");
const results = document.getElementById("results");
const overviewView = document.getElementById("overview");
const overviewResult = document.querySelector(".overview-result");
const statusMessage = document.getElementById("statusMessage");
const downloadBtn = document.getElementById("downloadBtn");

let latestReport = null;
let analysisRequestVersion = 0;

// Safely convert values to displayable text.
function text(value, fallback = "Not available") {
    if (value === null || value === undefined || value === "") {
        return fallback;
    }

    if (typeof value === "object") {
        return fallback;
    }

    return String(value);
}

function isRecord(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
}

// Create DOM elements without inserting untrusted HTML.
function makeElement(tag, className, content) {
    const element = document.createElement(tag);

    if (className) element.className = className;
    if (content !== undefined) element.textContent = text(content);

    return element;
}

function renderTags(containerId, items) {
    const container = document.getElementById(containerId);
    container.replaceChildren();

    if (!Array.isArray(items) || items.length === 0) {
        container.append(makeElement("span", "tag", "Not available"));
        return;
    }

    items.forEach(item => {
        container.append(makeElement("span", "tag", item));
    });
}

function renderList(containerId, items) {
    const container = document.getElementById(containerId);
    container.replaceChildren();

    if (!Array.isArray(items) || items.length === 0) {
        container.append(makeElement("li", "", "Not available"));
        return;
    }

    items.forEach(item => {
        container.append(makeElement("li", "", item));
    });
}

function renderKeywords(containerId, items) {
    const container = document.getElementById(containerId);
    container.replaceChildren();

    if (!Array.isArray(items) || items.length === 0) {
        container.append(makeElement("div", "keyword-item", "Not available"));
        return;
    }

    items.forEach(item => {
        container.append(makeElement("div", "keyword-item", item));
    });
}

function showError(message) {
    errorPanel.textContent = message;
    errorPanel.classList.remove("hidden");
}

function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    buttonText.textContent = isLoading ? "Analyzing..." : "Analyze Website";
    loadingPanel.classList.toggle("hidden", !isLoading);

    if (isLoading) {
        errorPanel.classList.add("hidden");
        results.classList.add("hidden");
        statusMessage.textContent = "✦ Connecting to the AI analysis engine...";
    }
}

function getWebsiteUrlValidationMessage(value) {
    const url = String(value || "").trim();
    if (!url) return "Please enter a website URL.";

    try {
        const parsed = new URL(url);
        if (["http:", "https:"].includes(parsed.protocol) && parsed.hostname) {
            return "";
        }
    } catch {
        // Fall through to the same clear message used for malformed URLs.
    }

    return "Enter a valid HTTP or HTTPS website URL, such as https://example.com.";
}

function handleUrlInputChange() {
    const wasLoading = analyzeBtn.disabled;
    const hadResults = latestReport !== null
        || !results.classList.contains("hidden")
        || statusMessage.textContent === "✓ Analysis completed successfully.";
    const hadError = !errorPanel.classList.contains("hidden");
    const validationMessage = getWebsiteUrlValidationMessage(urlInput.value);

    urlInput.setCustomValidity(validationMessage);
    analysisRequestVersion += 1;
    latestReport = null;
    results.classList.add("hidden");
    errorPanel.classList.add("hidden");

    if (wasLoading) setLoading(false);

    if (validationMessage) {
        showError(validationMessage);
        statusMessage.textContent = "Please correct the website URL before analysis.";
    } else if (wasLoading || hadResults || hadError) {
        statusMessage.textContent = "URL changed. Submit it to analyze the updated website.";
    }
}

urlInput.addEventListener("input", handleUrlInputChange);
urlInput.addEventListener("invalid", () => {
    const message = getWebsiteUrlValidationMessage(urlInput.value);
    urlInput.setCustomValidity(message);
    showError(message);
    statusMessage.textContent = "Please correct the website URL before analysis.";
});

function renderWebsite(website) {
    document.getElementById("websiteTitle").textContent =
        text(website.title, "Untitled Website");

    document.getElementById("websiteDescription").textContent =
        text(website.meta_description);

    const link = document.getElementById("websiteLink");
    const websiteUrl = text(website.url, "");

    link.textContent = websiteUrl;
    link.href = websiteUrl || "#";
}

function renderCompetitors(data) {
    const section = data.competitor_analysis || {};

    document.getElementById("brandName").textContent =
        text(section.brand_name);

    document.getElementById("brandCategory").textContent =
        text(section.category);

    document.getElementById("brandOverview").textContent =
        text(section.overview);

    renderTags("advantagesList", section.brand_advantages);

    const container = document.getElementById("competitorList");
    container.replaceChildren();

    const competitors = Array.isArray(section.competitors)
        ? section.competitors
        : [];

    if (competitors.length === 0) {
        container.append(
            makeElement("p", "muted", "No AI-suggested competitor claims available.")
        );
        return;
    }

    competitors.forEach(competitor => {
        const card = makeElement("article", "competitor-card");
        card.append(makeElement("h4", "", competitor.name));

        card.append(makeElement("strong", "", "STRENGTHS"));

        const strengths = makeElement("ul");
        (Array.isArray(competitor.strengths) ? competitor.strengths : [])
            .forEach(item => strengths.append(makeElement("li", "", item)));

        if (!strengths.children.length) {
            strengths.append(makeElement("li", "", "Not available"));
        }

        card.append(strengths);

        card.append(makeElement("strong", "weakness-title", "WEAKNESSES"));

        const weaknesses = makeElement("ul");
        (Array.isArray(competitor.weaknesses) ? competitor.weaknesses : [])
            .forEach(item => weaknesses.append(makeElement("li", "", item)));

        if (!weaknesses.children.length) {
            weaknesses.append(makeElement("li", "", "Not available"));
        }

        card.append(weaknesses);
        container.append(card);
    });
}


function formatScore(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "Not available";
    return `${number.toFixed(2)} / 1.00`;
}

function safeHttpUrl(value) {
    if (typeof value !== "string" || !value.trim()) return "";
    try {
        const parsed = new URL(value);
        return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : "";
    } catch {
        return "";
    }
}

function statusLabel(value) {
    return String(value || "not_available")
        .replaceAll("_", " ")
        .replace(/\b\w/g, character => character.toUpperCase());
}

function renderDiscoveredCompetitors(items) {
    const container = document.getElementById("discoveredCompetitorList");
    const summary = document.getElementById("discoverySummary");
    container.replaceChildren();

    const competitors = Array.isArray(items) ? items : [];
    summary.textContent = `${competitors.length} potential competitor${competitors.length === 1 ? "" : "s"} discovered`;

    if (!competitors.length) {
        container.append(makeElement(
            "p",
            "muted discovery-empty",
            "No potential competitors were returned. Check the Tavily configuration or try another website."
        ));
        return;
    }

    competitors.forEach((competitor, index) => {
        const card = makeElement("article", "discovered-card");
        const header = makeElement("div", "discovered-card-header");
        const titleGroup = makeElement("div", "discovered-title-group");
        titleGroup.append(makeElement("span", "competitor-rank", `#${index + 1}`));
        titleGroup.append(makeElement("h3", "", competitor.name || "Unnamed competitor"));
        header.append(titleGroup);

        const typeClass = competitor.competitor_type === "direct"
            ? "type-direct"
            : competitor.competitor_type === "indirect"
                ? "type-indirect"
                : "type-unclear";
        const type = makeElement(
            "span",
            `type-pill ${typeClass}`,
            `Evidence label: ${statusLabel(competitor.competitor_type || "unclear")}`
        );
        header.append(type);
        card.append(header);

        if (competitor.description) {
            card.append(makeElement("p", "discovered-description", competitor.description));
        }
        if (competitor.relevance_reason) {
            card.append(makeElement("p", "candidate-relevance-reason", `Why relevant: ${competitor.relevance_reason}`));
        }
        if (competitor.competitor_type_reason) {
            card.append(makeElement("p", "candidate-type-reason", competitor.competitor_type_reason));
        }
        if (
            competitor.competitor_type === "unclear"
            && ["direct", "indirect"].includes(competitor.ai_competitor_type)
        ) {
            card.append(makeElement(
                "p",
                "candidate-ai-type-note",
                `Gemini suggested ${competitor.ai_competitor_type}; accessible page evidence did not establish that classification.`
            ));
        }
        if (competitor.product_name) {
            card.append(makeElement("p", "discovered-product", `Product: ${competitor.product_name}`));
        }

        const metrics = makeElement("div", "discovery-metrics");
        const similarity = makeElement("div", "discovery-metric");
        similarity.append(makeElement("span", "metric-label", "Semantic similarity score"));
        similarity.append(makeElement("strong", "metric-value", formatScore(competitor.similarity_score)));
        const evidence = makeElement("div", "discovery-metric");
        evidence.append(makeElement("span", "metric-label", "Heuristic evidence score"));
        evidence.append(makeElement(
            "strong",
            "metric-value",
            formatScore(competitor.heuristic_evidence_score ?? competitor.evidence_confidence)
        ));
        metrics.append(similarity, evidence);
        card.append(metrics);

        const statusRow = makeElement("div", "evidence-status-row");
        statusRow.append(makeElement(
            "span",
            `evidence-pill ${String(competitor.evidence_strength || "").toLowerCase() === "strong" ? "evidence-strong" : "evidence-limited"}`,
            `Strength: ${statusLabel(competitor.evidence_strength || "not available")}`
        ));
        statusRow.append(makeElement(
            "span",
            "status-text",
            `Status: ${statusLabel(competitor.evidence_status)}`
        ));
        card.append(statusRow);

        const checked = Number(competitor.evidence_pages_checked);
        const accessible = Number(competitor.evidence_pages_accessible);
        if (Number.isFinite(checked) || Number.isFinite(accessible)) {
            const pageText = makeElement(
                "p",
                "evidence-page-count",
                `Pages checked: ${Number.isFinite(checked) ? checked : "—"} · Accessible: ${Number.isFinite(accessible) ? accessible : "—"}`
            );
            card.append(pageText);
        }

        const website = safeHttpUrl(competitor.official_website || competitor.url);
        if (website) {
            const websiteLink = makeElement("a", "source-link", "Visit website ↗");
            websiteLink.href = website;
            websiteLink.target = "_blank";
            websiteLink.rel = "noopener noreferrer";
            card.append(websiteLink);
        }

        const sourceUrls = Array.isArray(competitor.evidence_urls) ? competitor.evidence_urls : [];
        const details = Array.isArray(competitor.evidence_details) ? competitor.evidence_details : [];
        const detailUrls = details.map(detail => detail && detail.url).filter(Boolean);
        const allSources = [...new Set([...sourceUrls, ...detailUrls])]
            .map(safeHttpUrl)
            .filter(Boolean);

        if (allSources.length) {
            const sources = makeElement("div", "evidence-sources");
            sources.append(makeElement("span", "metric-label", "Evidence sources"));
            allSources.slice(0, 5).forEach((url, sourceIndex) => {
                const link = makeElement("a", "", `Source ${sourceIndex + 1} ↗`);
                link.href = url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                sources.append(link);
            });
            card.append(sources);
        }

        if (details.length) {
            const disclosure = document.createElement("details");
            disclosure.className = "evidence-details";
            const disclosureTitle = document.createElement("summary");
            disclosureTitle.textContent = `Observed webpage evidence (${details.length} page${details.length === 1 ? "" : "s"})`;
            disclosure.append(disclosureTitle);

            details.slice(0, 8).forEach(detail => {
                const page = makeElement("div", "evidence-page");
                const sourceUrl = safeHttpUrl(detail.source_url || detail.url);
                const sourceTypeLabels = {
                    official_website: "Official website",
                    official_website_candidate: "Candidate official site (unverified)",
                    third_party: "Third-party source",
                    unverified_source: "Unverified source"
                };
                const sourceType = sourceTypeLabels[detail.source_type]
                    || (detail.is_official_domain ? "Official website" : "Third-party source");
                const pageHeading = makeElement(
                    "p",
                    "evidence-page-heading",
                    `${detail.accessibility_status === "accessible" ? "Accessible" : "Inaccessible"} · ${sourceType} · ${statusLabel(detail.classification || "no_relevant_evidence")}`
                );
                page.append(pageHeading);
                if (sourceUrl) {
                    const pageLink = makeElement("a", "evidence-page-link", detail.source_url || detail.url);
                    pageLink.href = sourceUrl;
                    pageLink.target = "_blank";
                    pageLink.rel = "noopener noreferrer";
                    page.append(pageLink);
                }
                if (detail.reason) {
                    page.append(makeElement("p", "evidence-reason", detail.reason));
                }

                const excerpts = Array.isArray(detail.excerpts) ? detail.excerpts : [];
                if (!excerpts.length && detail.accessibility_status === "accessible") {
                    const duplicateCount = Number(detail.duplicate_excerpts_omitted || 0);
                    page.append(makeElement(
                        "p",
                        "evidence-no-match",
                        duplicateCount > 0
                            ? "A repeated excerpt was omitted; matching text is shown with another source above."
                            : "No relevant text match was found on this accessible page."
                    ));
                }
                excerpts.forEach(excerpt => {
                    const excerptBlock = makeElement("blockquote", "evidence-excerpt");
                    excerptBlock.append(makeElement(
                        "p",
                        "evidence-excerpt-text",
                        excerpt.matched_text || ""
                    ));
                    excerptBlock.append(makeElement(
                        "p",
                        "evidence-excerpt-meta",
                        `Competitor: ${excerpt.competitor_name || competitor.name || "Unknown"} · ${statusLabel(excerpt.classification || "observed_text")}`
                    ));
                    const terms = Array.isArray(excerpt.matching_terms) ? excerpt.matching_terms : [];
                    if (terms.length) {
                        excerptBlock.append(makeElement(
                            "p",
                            "evidence-excerpt-terms",
                            `Matched terms: ${terms.join(", ")}`
                        ));
                    }
                    page.append(excerptBlock);
                });
                disclosure.append(page);
            });
            card.append(disclosure);
        }

        card.append(makeElement(
            "p",
            "discovery-disclaimer",
            "Quoted text is observed webpage content. Direct/indirect labels require explicit relationship wording on a fetched page; all matching scores remain heuristic signals, not probabilities or proof that businesses compete."
        ));
        container.append(card);
    });
}

function renderSEO(data) {
    const seo = data.seo_analysis || {};
    const keywordIntents = seo.keyword_intents || {};

    renderKeywordGroup("primaryKeywords", seo.primary_keywords, keywordIntents.primary, true);
    renderKeywordGroup("secondaryKeywords", seo.secondary_keywords, keywordIntents.secondary, true);
    renderKeywordGroup("longTailKeywords", seo.long_tail_keywords, keywordIntents.long_tail, false);
    renderKeywordGroup("businessKeywords", seo.business_keywords, keywordIntents.business, true);
    renderKeywordGroup("brandedProductKeywords", seo.branded_product_keywords, keywordIntents.branded_products, true);
    renderMetadataAnalysis(seo.metadata_assessment);
    renderOnPageSeoScore(seo.seo_score);
    renderContentGaps(seo.content_gaps);

    const legacyRecommendations = document.getElementById("seoRecommendations");
    const prioritized = Array.isArray(seo.prioritized_recommendations)
        ? seo.prioritized_recommendations
        : [];
    const prioritizedContainer = document.getElementById("prioritizedSeoRecommendations");
    prioritizedContainer.replaceChildren();
    if (prioritized.length) {
        legacyRecommendations.classList.add("hidden");
        prioritized.forEach(item => {
            const card = makeElement("div", "keyword-item");
            card.append(makeElement("strong", "", `${text(item.priority, "Medium")} priority`));
            card.append(makeElement("p", "", item.recommendation));
            card.append(makeElement("p", "muted", `Expected benefit: ${text(item.expected_benefit)}`));
            card.append(makeElement("p", "muted", `Implementation effort: ${text(item.implementation_effort)}`));
            const relatedLabel = item.related_to === "content_gap"
                ? `Content opportunity: ${text(item.related_topic)}`
                : `Observed issue: ${statusLabel(item.related_to || "not available")}`;
            card.append(makeElement("p", "muted", relatedLabel));
            prioritizedContainer.append(card);
        });
        legacyRecommendations.replaceChildren();
    } else {
        legacyRecommendations.classList.remove("hidden");
        renderList("seoRecommendations", seo.content_recommendations);
    }

    const table = document.getElementById("queryTable");
    table.replaceChildren();

    const queries = Array.isArray(data.target_search_queries)
        ? data.target_search_queries
        : [];

    queries.forEach(item => {
        const row = document.createElement("tr");

        [
            item.query,
            `${text(item.platform)} · ${statusLabel(item.platform_type || "not available")}`,
            item.intent,
            item.relevance
        ].forEach((value, index) => {
            const cell = document.createElement("td");

            if (index === 3) {
                cell.append(makeElement("span", "relevance", value));
            } else {
                cell.textContent = text(value);
            }

            row.append(cell);
        });

        table.append(row);
    });

    if (!queries.length) {
        const row = document.createElement("tr");
        const cell = makeElement("td", "", "No search queries available.");
        cell.colSpan = 4;
        row.append(cell);
        table.append(row);
    }
}

function renderKeywordGroup(containerId, keywords, intentEntries, useTags) {
    if (!Array.isArray(intentEntries)) {
        if (useTags) renderTags(containerId, keywords);
        else renderKeywords(containerId, keywords);
        return;
    }

    const container = document.getElementById(containerId);
    container.replaceChildren();
    if (!Array.isArray(keywords) || keywords.length === 0) {
        container.append(makeElement(useTags ? "span" : "div", useTags ? "tag" : "keyword-item", "Not available"));
        return;
    }

    const intents = new Map(intentEntries
        .filter(item => item && typeof item.keyword === "string")
        .map(item => [item.keyword.toLocaleLowerCase(), item.intent]));
    keywords.forEach(keyword => {
        const intent = intents.get(String(keyword).toLocaleLowerCase());
        const label = intent
            ? `${keyword} · Intent: ${statusLabel(intent)}`
            : keyword;
        container.append(makeElement(useTags ? "span" : "div", useTags ? "tag" : "keyword-item", label));
    });
}

function renderMetadataAnalysis(metadata) {
    const container = document.getElementById("metadataAnalysis");
    container.replaceChildren();
    if (!metadata || typeof metadata !== "object") {
        container.append(makeElement("p", "muted", "Metadata assessment unavailable."));
        return;
    }

    [
        ["Title tag", metadata.title_tag],
        ["Meta description", metadata.meta_description],
    ].forEach(([label, item]) => {
        const block = makeElement("div", "keyword-item");
        if (!item || typeof item !== "object") {
            block.append(makeElement("strong", "", `${label}: unavailable`));
        } else {
            block.append(makeElement("strong", "", `${label} · ${statusLabel(item.status)}`));
            block.append(makeElement("p", "", `Observed (${text(item.character_count, "0")} characters): ${text(item.observed, "Missing")}`));
            block.append(makeElement("p", "muted", `AI suggestion: ${text(item.suggestion)}`));
        }
        container.append(block);
    });
}

function renderOnPageSeoScore(score) {
    const container = document.getElementById("seoScore");
    container.replaceChildren();
    if (!score || !Array.isArray(score.components)) {
        container.append(makeElement("p", "muted", "Score unavailable."));
        return;
    }

    container.append(makeElement("h3", "", `${text(score.score, "0")} / ${text(score.max_score, "100")}`));
    container.append(makeElement("p", "muted", score.methodology));
    score.components.forEach(component => {
        container.append(makeElement(
            "div",
            "keyword-item",
            `${component.factor}: ${component.score}/${component.max_score} · ${component.finding}`
        ));
    });
}

function renderContentGaps(items) {
    const container = document.getElementById("contentGaps");
    container.replaceChildren();
    if (!Array.isArray(items) || !items.length) {
        container.append(makeElement("p", "muted", "No content gaps were suggested from the available page content."));
        return;
    }

    items.forEach(item => {
        const card = makeElement("div", "keyword-item");
        card.append(makeElement("strong", "", item.topic));
        card.append(makeElement("p", "", `Why it matters: ${text(item.why_it_matters)}`));
        card.append(makeElement("p", "muted", `Opportunity: ${text(item.content_opportunity)}`));
        container.append(card);
    });
}

function renderServices(data) {
    const container = document.getElementById("servicesGrid");
    container.replaceChildren();

    const services = Array.isArray(data.top_services)
        ? data.top_services
        : [];

    if (!services.length) {
        container.append(
            makeElement("p", "muted", "No services available.")
        );
        return;
    }

    services.forEach((service, index) => {
        const card = makeElement("article", "service-card");

        card.append(makeElement("div", "service-icon", "◇"));
        card.append(makeElement("h3", "", service.name));
        card.append(makeElement("p", "", service.description));

        card.append(
            makeElement("div", "service-meta",
                "Industry: " + text(service.target_industry))
        );

        card.append(
            makeElement("div", "service-meta",
                "Customer: " + text(service.target_customer))
        );

        card.append(
            makeElement("div", "service-meta",
                "Benefit: " + text(service.business_benefit))
        );

        if (service.inferred === true) {
            card.append(
                makeElement("span", "inferred-label", "AI-INFERRED SUGGESTION")
            );
        }

        container.append(card);
    });
}

function renderAudience(data) {
    const audience = data.target_audience || {};

    document.getElementById("primaryAudience").textContent =
        text(audience.primary_audience);

    document.getElementById("secondaryAudience").textContent =
        text(audience.secondary_audience);

    renderTags("industrySegments", audience.industry_segments);
    renderTags("businessTypes", audience.business_types);
    renderList("customerProfiles", audience.customer_profiles);
    renderList("customerNeeds", audience.needs_and_challenges);

    const evidenceContainer = document.getElementById("audienceEvidence");
    evidenceContainer.replaceChildren();
    const evidenceItems = [
        ...(Array.isArray(audience.industry_segment_evidence) ? audience.industry_segment_evidence : [])
            .map(item => ({ ...item, field: "industry_segments", claim: item.industry })),
        ...(Array.isArray(audience.audience_evidence) ? audience.audience_evidence : []),
    ];
    if (!evidenceItems.length) {
        evidenceContainer.append(makeElement("p", "muted", "No audience or industry claims had sufficient extracted-page support."));
        return;
    }
    evidenceItems.forEach(item => {
        const card = makeElement("div", "keyword-item");
        const basis = item.basis === "observed" ? "Observed on page" : "AI inference";
        card.append(makeElement("strong", "", `${text(item.claim)} · ${basis}`));
        card.append(makeElement("p", "muted", `Field: ${statusLabel(item.field)}`));
        card.append(makeElement("blockquote", "", item.supporting_text));
        if (item.basis === "inferred" && item.reason) {
            card.append(makeElement("p", "muted", `Inference: ${item.reason}`));
        }
        evidenceContainer.append(card);
    });
}

function renderReport(response) {
    const website = response.website || {};
    const analysis = response.analysis || {};

    renderWebsite(website);
    renderCompetitors(analysis);
    renderDiscoveredCompetitors(analysis.discovered_competitors);
    renderSEO(analysis);
    renderServices(analysis);
    renderAudience(analysis);

    latestReport = response;

    setDashboardView("overview");
    errorPanel.classList.add("hidden");
    statusMessage.textContent = "✓ Analysis completed successfully.";

    overviewView.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

function setDashboardView(viewId) {
    const isOverview = viewId === "overview";
    document.querySelectorAll("#results .content-section").forEach(section => {
        section.classList.add("hidden");
    });

    if (isOverview) {
        overviewView.classList.remove("hidden");
        results.classList.toggle("hidden", !latestReport);
        overviewResult.classList.remove("hidden");
        return;
    }

    const selectedSection = document.getElementById(viewId);
    if (!selectedSection || !results.contains(selectedSection)) return;

    overviewView.classList.add("hidden");
    overviewResult.classList.add("hidden");
    results.classList.remove("hidden");
    selectedSection.classList.remove("hidden");
}

form.addEventListener("submit", async event => {
    event.preventDefault();

    const url = urlInput.value.trim();
    const validationMessage = getWebsiteUrlValidationMessage(url);

    if (validationMessage) {
        urlInput.setCustomValidity(validationMessage);
        showError(validationMessage);
        statusMessage.textContent = "Please correct the website URL before analysis.";
        return;
    }

    urlInput.setCustomValidity("");
    const requestVersion = ++analysisRequestVersion;
    latestReport = null;
    setLoading(true);

    try {
        let response;
        try {
            response = await fetch(API_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify({ url })
            });
        } catch (error) {
            const networkError = new Error(
                "Could not connect to the API. Check your internet connection and API CORS settings."
            );
            networkError.name = "NetworkRequestError";
            throw networkError;
        }

        if (requestVersion !== analysisRequestVersion) return;

        let responseText;
        try {
            responseText = await response.text();
        } catch (error) {
            throw new Error(`Could not read the API response (HTTP ${response.status}).`);
        }

        let data = null;
        if (!responseText.trim() && response.ok) {
            throw new Error(`The API returned an empty response (HTTP ${response.status}).`);
        }
        if (responseText.trim()) {
            try {
                data = JSON.parse(responseText);
            } catch (error) {
                if (response.ok) {
                    throw new Error(`The API returned invalid JSON (HTTP ${response.status}).`);
                }
            }
        }

        if (!response.ok) {
            const detail = data?.error?.message || data?.detail;
            const message = typeof detail === "string"
                ? `API request failed (HTTP ${response.status}): ${detail}`
                : `API request failed with HTTP ${response.status}.`;

            throw new Error(message);
        }

        if (
            !isRecord(data)
            || data.success !== true
            || !isRecord(data.website)
            || typeof data.website.url !== "string"
            || typeof data.website.title !== "string"
            || typeof data.website.meta_description !== "string"
            || !isRecord(data.analysis)
            || !isRecord(data.analysis.competitor_analysis)
            || !Array.isArray(data.analysis.target_search_queries)
            || !isRecord(data.analysis.seo_analysis)
            || !Array.isArray(data.analysis.top_services)
            || !isRecord(data.analysis.target_audience)
        ) {
            throw new Error("The API returned an unexpected response format.");
        }

        try {
            renderReport(data);
        } catch (error) {
            results.classList.add("hidden");
            latestReport = null;
            console.error("Analysis dashboard rendering failed:", error);
            throw new Error("The analysis succeeded, but the dashboard could not display the results. Reload the page and try again.");
        }

    } catch (error) {
        if (requestVersion !== analysisRequestVersion) return;
        console.error("Website analysis failed:", error);

        let message = error.message || "Something went wrong.";

        showError(message);
        statusMessage.textContent = "Analysis could not be completed.";

    } finally {
        if (requestVersion === analysisRequestVersion) setLoading(false);
    }
});

// Download the exact API response as a JSON report.
downloadBtn.addEventListener("click", () => {
    if (!latestReport) {
        alert("Please analyze a website before exporting a report.");
        return;
    }

    const blob = new Blob(
        [JSON.stringify(latestReport, null, 2)],
        { type: "application/json" }
    );

    const downloadUrl = URL.createObjectURL(blob);
    const anchor = document.createElement("a");

    const brand = latestReport.analysis?.competitor_analysis?.brand_name
        || "website-analysis";

    anchor.href = downloadUrl;
    anchor.download = `${brand.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-report.json`;

    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();

    URL.revokeObjectURL(downloadUrl);
});

// Switch dashboard views and keep the sidebar in sync.
document.querySelectorAll(".navigation .nav-link").forEach(link => {
    link.addEventListener("click", event => {
        const viewId = link.hash.slice(1);
        const target = document.getElementById(viewId);
        if (!target) return;

        event.preventDefault();

        document.querySelectorAll(".navigation .nav-link")
            .forEach(item => item.classList.remove("active"));
        link.classList.add("active");
        setDashboardView(viewId);

        history.replaceState(null, "", link.hash);
        target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
});
