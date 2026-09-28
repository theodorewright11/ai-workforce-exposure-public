"use client";

/* Guide — the two measures, how to read them, and what they are not.
 *
 * Deliberately short. The dashboard's voice is plain numbers with minimal
 * interpretation (PRD §1.4); this page carries the framing so the other
 * pages don't have to.
 */

const SECTION: React.CSSProperties = {
  background: "var(--bg-surface)", border: "1px solid var(--border)",
  borderRadius: 10, padding: "20px 24px", marginBottom: 18,
};
const H2: React.CSSProperties = {
  fontSize: 16, fontWeight: 700, color: "var(--text-primary)", marginBottom: 10,
};
const P: React.CSSProperties = {
  fontSize: 13.5, color: "var(--text-secondary)", lineHeight: 1.68, marginBottom: 12,
};
const EQ: React.CSSProperties = {
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 12,
  background: "var(--brand-light)", borderRadius: 6, padding: "10px 14px",
  color: "var(--text-primary)", marginBottom: 12, overflowX: "auto", whiteSpace: "pre",
};

export default function GuidePage() {
  return (
    <div style={{ maxWidth: 780, margin: "0 auto", padding: "28px 24px 72px" }}>
      <h1 style={{ fontSize: 25, fontWeight: 700, color: "var(--text-primary)", marginBottom: 20 }}>Guide</h1>

      <div style={SECTION}>
        <div style={H2}>The one question</div>
        <p style={P}>
          Everything here answers a single question: <strong>what work can AI actually do, and
          how does that differ by job and by field?</strong> Two measurements bear on it, and
          every number on this site is one of the two.
        </p>
      </div>

      <div style={SECTION}>
        <div style={H2}>1 · Exposure — the share of the workday</div>
        <p style={P}>
          Each task an occupation performs carries an estimate of how many hours a day it takes,
          normalised so a job&rsquo;s tasks add up to a seven-hour day. A task counts toward
          exposure if people have brought it to AI often enough to pass a minimum threshold in
          the usage data, and it counts in proportion to the level of automation they brought
          it at.
        </p>
        <div style={EQ}>{`exposed hours = hours per day × workers × (automation level ÷ 5)
exposure %    = exposed hours ÷ total hours`}</div>
        <p style={P}>
          <strong>Exposure is not the share of work AI is doing today.</strong> It answers a
          what-if: if every worker used AI on the tasks people have already brought to it, at
          the level of automation they brought them at, this is the share of work time AI would
          do or assist. Most workers do not use AI that way yet, so the real share is lower.
        </p>
        <p style={P}>
          So <strong>62% exposed means 62% of the workday under that what-if</strong>, not 62% of
          tasks and not 62% of jobs. Above a single occupation the percentage is
          employment-weighted: &ldquo;of all the hours worked in this group, this share is
          exposed.&rdquo;
        </p>
      </div>

      <div style={SECTION}>
        <div style={H2}>2 · Usage — where AI actually gets used</div>
        <p style={P}>
          How much real AI activity lands on that work, per worker doing it, shown as a multiple
          of the median. <strong>1.00×</strong> is the median group; 5× is five times that.
        </p>
        <p style={P}>
          Usage is mainly a <strong>second signal about capability</strong> rather than a measure
          of rollout. People don&rsquo;t repeatedly bring a task to AI that AI fails at, so heavy
          use is corroborating evidence that the work really is reachable. When exposure and usage
          agree, the reading is strong; when they disagree, it&rsquo;s weak.
        </p>
        <p style={{ ...P, marginBottom: 0 }}>
          Usage divides by employment alone, not by hours — it asks &ldquo;per worker doing this
          work, how much AI activity is there,&rdquo; and re-weighting by task length would
          penalise slow tasks.
        </p>
      </div>

      <div style={SECTION}>
        <div style={H2}>Why the bars always show two things</div>
        <p style={{ ...P, marginBottom: 0 }}>
          A single number invites panic and supports no decision. Every bar on this site draws the
          work AI reaches <em>and</em> the work it doesn&rsquo;t, because the second half is where
          the useful information is: which parts of a job are moving, and which are not.
        </p>
      </div>

      <div style={SECTION}>
        <div style={H2}>Verb families</div>
        <p style={{ ...P, marginBottom: 0 }}>
          Tasks grouped by the kind of action they are — Analyze, Create, Document, Evaluate,
          Manage, and so on — cutting across occupations. This is a different cut from O*NET&rsquo;s
          work-activity hierarchy, which groups by domain. It matters because exposure varies far
          more between kinds of work than between fields: analysis and documentation are heavily
          reachable in almost every occupation, while evaluation and management are not.
        </p>
      </div>

      <div style={SECTION}>
        <div style={H2}>What this is not</div>
        <p style={P}>
          <strong>Not a job-loss forecast.</strong> Exposure measures overlap with observed AI
          capability. High exposure does not mean an occupation disappears — it means a large
          share of its work is technically reachable today.
        </p>
        <p style={P}>
          <strong>It is an upper bound.</strong> Exposure assumes every worker uses AI on every
          task it has been observed on, at the level of automation observed. Most do not, so it
          runs ahead of what is actually happening in workplaces.
        </p>
        <p style={P}>
          <strong>Growth is partly measurement.</strong> Some of the rise across snapshots is real
          capability and adoption growth; some is this dataset becoming more complete over time,
          which is a one-time gain that flattens. Read projections as a ceiling.
        </p>
        <p style={{ ...P, marginBottom: 0 }}>
          <strong>Physical work is under-covered by construction.</strong> The underlying data is
          digital AI use, so physical and robotic work is measured only where it shows up in that
          record. It is shown separately rather than hidden inside the aggregate.
        </p>
      </div>

      <div style={SECTION}>
        <div style={H2}>Sources</div>
        <p style={P}>
          Real-world AI usage from Anthropic&rsquo;s Claude (conversational and API) and
          Microsoft&rsquo;s Copilot; an MCP-server capability pipeline; occupation and task
          structure from O*NET; employment and wages from BLS OEWS; employment projections from
          the BLS 2025&ndash;34 outlook. U.S. only.
        </p>
        <p style={{ ...P, marginBottom: 0 }}>
          <strong style={{ color: "var(--brand)" }}>The paper →</strong>{" "}
          <span style={{ color: "var(--text-muted)" }}>
            Mapping AI Exposure Across the U.S. Workforce: Evidence from Millions of AI
            Conversations (Wright, Schwarze &amp; Boyd, 2026) — coming soon
          </span>
        </p>
      </div>

      <p style={{ fontSize: 12, color: "var(--text-muted)", lineHeight: 1.6 }}>
        A project of Utah&rsquo;s Office of AI Policy (OAIP), supported by the BYU Department of
        Mathematics.
      </p>
    </div>
  );
}
