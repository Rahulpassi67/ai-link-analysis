const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const appSource = fs.readFileSync(
    path.join(__dirname, "..", "frontend", "app.js"),
    "utf8"
);

class FakeClassList {
    constructor(initial = []) {
        this.values = new Set(initial);
    }

    add(value) { this.values.add(value); }
    remove(value) { this.values.delete(value); }
    contains(value) { return this.values.has(value); }
    toggle(value, force) {
        const shouldAdd = force === undefined ? !this.contains(value) : force;
        if (shouldAdd) this.add(value);
        else this.remove(value);
        return shouldAdd;
    }
}

class FakeElement {
    constructor(id = "") {
        this.id = id;
        this.classList = new FakeClassList();
        this.listeners = {};
        this.children = [];
        this.textContent = "";
        this.value = "";
        this.disabled = false;
        this.validityMessage = "";
    }

    addEventListener(name, handler) { this.listeners[name] = handler; }
    setCustomValidity(message) { this.validityMessage = message; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    scrollIntoView() {}
    click() {}
    remove() {}
}

function makeHarness(fetchImpl) {
    const elements = new Map();
    const document = {
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, new FakeElement(id));
            return elements.get(id);
        },
        createElement(tag) { return new FakeElement(tag); },
        querySelectorAll() { return []; },
        body: new FakeElement("body"),
    };
    for (const id of ["results", "loadingPanel", "errorPanel"]) {
        document.getElementById(id).classList.add("hidden");
    }

    const context = vm.createContext({
        document,
        fetch: fetchImpl,
        URL,
        console: { error() {} },
    });
    vm.runInContext(appSource, context);
    return { elements, document, context };
}

function renderedText(element) {
    return [element.textContent, ...element.children.map(renderedText)].join(" ");
}

function successfulResponse() {
    return {
        ok: true,
        status: 200,
        async text() {
            return JSON.stringify({
                success: true,
                website: {
                    url: "https://shopify.com/",
                    title: "Shopify",
                    meta_description: "",
                },
                analysis: {
                    competitor_analysis: {},
                    target_search_queries: [],
                    seo_analysis: {},
                    top_services: [],
                    target_audience: {},
                    discovered_competitors: [],
                },
            });
        },
    };
}

function responseWith(status, body) {
    return {
        ok: status >= 200 && status < 300,
        status,
        async text() { return body; },
    };
}

async function submit(form) {
    await form.listeners.submit({ preventDefault() {} });
}

test("editing the URL hides the previous successful report", async () => {
    const calls = [];
    const { elements } = makeHarness(async (...args) => {
        calls.push(args);
        return successfulResponse();
    });
    const form = elements.get("analysisForm");
    const input = elements.get("websiteUrl");
    const results = elements.get("results");
    const status = elements.get("statusMessage");

    input.value = "https://shopify.com/";
    await submit(form);
    assert.equal(results.classList.contains("hidden"), false);
    assert.equal(status.textContent, "✓ Analysis completed successfully.");

    input.value = "https://example.com/";
    input.listeners.input();

    assert.equal(results.classList.contains("hidden"), true);
    assert.match(status.textContent, /URL changed/i);
    assert.notEqual(status.textContent, "✓ Analysis completed successfully.");
    assert.equal(calls.length, 1);
});

test("malformed URL shows validation and cannot reuse previous success state", async () => {
    let calls = 0;
    const { elements } = makeHarness(async () => {
        calls += 1;
        return successfulResponse();
    });
    const form = elements.get("analysisForm");
    const input = elements.get("websiteUrl");
    const results = elements.get("results");
    const error = elements.get("errorPanel");
    const status = elements.get("statusMessage");

    input.value = "https://shopify.com/";
    await submit(form);
    input.value = "not-a-valid-url";
    input.listeners.input();
    input.listeners.invalid();
    await submit(form);

    assert.equal(results.classList.contains("hidden"), true);
    assert.equal(error.classList.contains("hidden"), false);
    assert.match(error.textContent, /valid HTTP or HTTPS website URL/i);
    assert.match(status.textContent, /correct the website URL/i);
    assert.notEqual(status.textContent, "✓ Analysis completed successfully.");
    assert.equal(calls, 1);
});

test("response from an analysis becomes stale after the URL changes", async () => {
    let resolveFetch;
    const { elements } = makeHarness(() => new Promise(resolve => {
        resolveFetch = resolve;
    }));
    const form = elements.get("analysisForm");
    const input = elements.get("websiteUrl");
    const results = elements.get("results");
    const status = elements.get("statusMessage");

    input.value = "https://shopify.com/";
    const pending = submit(form);
    input.value = "not-a-valid-url";
    input.listeners.input();
    resolveFetch(successfulResponse());
    await pending;

    assert.equal(results.classList.contains("hidden"), true);
    assert.match(status.textContent, /correct the website URL/i);
    assert.notEqual(status.textContent, "✓ Analysis completed successfully.");
});

test("SEO renderer shows keyword intent, observed metadata, score and recommendations", () => {
    const { elements, context } = makeHarness(async () => successfulResponse());
    context.renderSEO({
        seo_analysis: {
            primary_keywords: ["inventory management software"],
            secondary_keywords: [],
            long_tail_keywords: [],
            business_keywords: [],
            branded_product_keywords: ["Shop Pay"],
            keyword_intents: {
                primary: [{ keyword: "inventory management software", intent: "commercial" }],
                secondary: [],
                long_tail: [],
                business: [],
                branded_products: [{ keyword: "Shop Pay", intent: "navigational" }],
            },
            metadata_assessment: {
                title_tag: {
                    observed: "Short title",
                    status: "weak",
                    character_count: 12,
                    suggestion: "A clearer suggested page title",
                },
                meta_description: {
                    observed: "",
                    status: "missing",
                    character_count: 0,
                    suggestion: "A suggested description for the page.",
                },
            },
            seo_score: {
                score: 55,
                max_score: 100,
                methodology: "Heuristic score from extracted page data.",
                components: [{
                    factor: "Title tag",
                    score: 15,
                    max_score: 25,
                    finding: "12 observed characters.",
                }],
            },
            content_gaps: [{
                topic: "Inventory onboarding",
                why_it_matters: "It answers a common setup question.",
                content_opportunity: "Publish a setup guide.",
            }],
            prioritized_recommendations: [{
                recommendation: "Add a descriptive page title.",
                priority: "High",
                expected_benefit: "Clarifies the page topic.",
                implementation_effort: "Low",
                related_to: "title_tag",
                related_topic: "",
            }],
        },
        target_search_queries: [{
            query: "How inventory tools work",
            platform: "Google",
            platform_type: "search_engine",
            intent: "informational",
            relevance: "Relevant to inventory management software.",
        }],
    });

    assert.match(renderedText(elements.get("primaryKeywords")), /Intent: Commercial/);
    assert.match(renderedText(elements.get("metadataAnalysis")), /Observed \(12 characters\): Short title/);
    assert.match(renderedText(elements.get("seoScore")), /55 \/ 100/);
    assert.match(renderedText(elements.get("contentGaps")), /Inventory onboarding/);
    assert.match(renderedText(elements.get("prioritizedSeoRecommendations")), /High priority/);
    assert.match(renderedText(elements.get("prioritizedSeoRecommendations")), /Observed issue: Title Tag/);
    assert.match(renderedText(elements.get("brandedProductKeywords")), /Shop Pay/);
    assert.match(renderedText(elements.get("queryTable")), /Google · Search Engine/);
    assert.equal(elements.get("seoRecommendations").classList.contains("hidden"), true);
});

test("network failures retain the connection message", async () => {
    const { elements } = makeHarness(async () => {
        throw new TypeError("Failed to fetch");
    });
    elements.get("websiteUrl").value = "https://shopify.com/";

    await submit(elements.get("analysisForm"));

    assert.match(elements.get("errorPanel").textContent, /Could not connect to the API/i);
});

test("HTTP errors are reported with status instead of as network failures", async () => {
    const { elements } = makeHarness(async () => responseWith(502, "Bad Gateway"));
    elements.get("websiteUrl").value = "https://shopify.com/";

    await submit(elements.get("analysisForm"));

    assert.match(elements.get("errorPanel").textContent, /HTTP 502/);
    assert.doesNotMatch(elements.get("errorPanel").textContent, /Could not connect/i);
});

test("invalid JSON from a successful HTTP response has a distinct error", async () => {
    const { elements } = makeHarness(async () => responseWith(200, "not JSON"));
    elements.get("websiteUrl").value = "https://shopify.com/";

    await submit(elements.get("analysisForm"));

    assert.match(elements.get("errorPanel").textContent, /invalid JSON \(HTTP 200\)/i);
});

test("rendering errors are not mislabeled as connection failures", async () => {
    const { elements, context } = makeHarness(async () => successfulResponse());
    context.renderReport = () => { throw new TypeError("renderer failure"); };
    elements.get("websiteUrl").value = "https://shopify.com/";

    await submit(elements.get("analysisForm"));

    assert.match(elements.get("errorPanel").textContent, /dashboard could not display/i);
    assert.doesNotMatch(elements.get("errorPanel").textContent, /Could not connect/i);
});

test("unexpected successful response shapes are reported explicitly", async () => {
    const { elements } = makeHarness(async () => responseWith(200, JSON.stringify({ success: true })));
    elements.get("websiteUrl").value = "https://shopify.com/";

    await submit(elements.get("analysisForm"));

    assert.match(elements.get("errorPanel").textContent, /unexpected response format/i);
});

test("audience view distinguishes observed quotes from AI inferences", () => {
    const { elements, context } = makeHarness(async () => successfulResponse());
    context.renderAudience({
        target_audience: {
            primary_audience: "Online merchants",
            secondary_audience: "Not available",
            industry_segments: ["Ecommerce"],
            industry_segment_evidence: [{
                industry: "Ecommerce",
                basis: "inferred",
                supporting_text: "Shopify ecommerce platform helps merchants sell online.",
                reason: "The page describes ecommerce platform services.",
            }],
            business_types: [],
            customer_profiles: [],
            needs_and_challenges: [],
            audience_evidence: [{
                field: "primary_audience",
                claim: "Online merchants",
                basis: "observed",
                supporting_text: "Shopify helps merchants sell online.",
                reason: "The page directly names merchants.",
            }],
        },
    });

    assert.match(renderedText(elements.get("audienceEvidence")), /Observed on page/);
    assert.match(renderedText(elements.get("audienceEvidence")), /AI inference/);
    assert.match(renderedText(elements.get("audienceEvidence")), /Shopify helps merchants sell online/);
});
