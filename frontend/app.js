
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
            makeElement("p", "muted", "No verified competitors available.")
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
