
const API_URL = "/link-analysis";
const form = document.getElementById("analysisForm");
const urlInput = document.getElementById("websiteUrl");
const analyzeBtn = document.getElementById("analyzeBtn");
const buttonText = document.getElementById("buttonText");
const loadingPanel = document.getElementById("loadingPanel");
const errorPanel = document.getElementById("errorPanel");
const results = document.getElementById("results");
const statusMessage = document.getElementById("statusMessage");
const downloadBtn = document.getElementById("downloadBtn");

let latestReport = null;

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

    renderTags("primaryKeywords", seo.primary_keywords);
    renderTags("secondaryKeywords", seo.secondary_keywords);
    renderKeywords("longTailKeywords", seo.long_tail_keywords);
    renderTags("businessKeywords", seo.business_keywords);
    renderList("seoRecommendations", seo.content_recommendations);

    const table = document.getElementById("queryTable");
    table.replaceChildren();

    const queries = Array.isArray(data.target_search_queries)
        ? data.target_search_queries
        : [];

    queries.forEach(item => {
        const row = document.createElement("tr");

        [
            item.query,
            item.platform,
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

    results.classList.remove("hidden");
    errorPanel.classList.add("hidden");
    statusMessage.textContent = "✓ Analysis completed successfully.";

    results.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

form.addEventListener("submit", async event => {
    event.preventDefault();

    const url = urlInput.value.trim();

    if (!url) {
        showError("Please enter a website URL.");
        return;
    }

    setLoading(true);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({ url })
        });

        const data = await response.json();

        if (!response.ok) {
            const detail = data.detail;
            const message = typeof detail === "string"
                ? detail
                : "The server rejected this request. Check the website URL and try again.";

            throw new Error(message);
        }

        if (data.success !== true || !data.analysis) {
            throw new Error("The API returned an unexpected response format.");
        }

        renderReport(data);

    } catch (error) {
        console.error("Website analysis failed:", error);

        let message = error.message || "Something went wrong.";

        if (error instanceof TypeError) {
            message = "Could not connect to the API. Check your internet connection and API CORS settings.";
        }

        showError(message);
        statusMessage.textContent = "Analysis could not be completed.";

    } finally {
        setLoading(false);
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

// Update the active sidebar item while navigating.
document.querySelectorAll(".navigation .nav-link").forEach(link => {
    link.addEventListener("click", () => {
        document.querySelectorAll(".navigation .nav-link")
            .forEach(item => item.classList.remove("active"));

        link.classList.add("active");
    });
});
