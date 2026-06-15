import { parseAnswer } from "../lib/parseAnswer";

describe("parseAnswer", () => {
  it("returns display text unchanged when no SUGGEST marker", () => {
    const result = parseAnswer("The ECE department is located at 301 WEB.");
    expect(result.display).toBe("The ECE department is located at 301 WEB.");
    expect(result.suggestions).toBeUndefined();
  });

  it("splits real follow-up suggestions correctly", () => {
    const raw =
      "Here is the info.|||SUGGEST: What programs are offered? | Who are the faculty? | How do I apply?";
    const result = parseAnswer(raw);
    expect(result.display).toBe("Here is the info.");
    expect(result.suggestions).toEqual([
      "What programs are offered?",
      "Who are the faculty?",
      "How do I apply?",
    ]);
  });

  it("filters out literal <q1> <q2> <q3> placeholders (regression: issue #12)", () => {
    const raw = "Some answer.|||SUGGEST: <q1> | <q2> | <q3>";
    const result = parseAnswer(raw);
    expect(result.suggestions).toBeUndefined();
  });

  it("filters out angle-bracket wrapped questions (regression: issue #11)", () => {
    const raw = "Some answer.|||SUGGEST: <What are the admission requirements?> | <Who are the faculty?> | <How do I apply?>";
    const result = parseAnswer(raw);
    expect(result.suggestions).toBeUndefined();
  });

  it("filters mixed real + placeholder suggestions, keeping only real ones", () => {
    const raw = "Answer.|||SUGGEST: <q1> | What is the GPA requirement? | <q3>";
    const result = parseAnswer(raw);
    expect(result.suggestions).toEqual(["What is the GPA requirement?"]);
  });

  it("keeps real questions that happen to contain angle brackets mid-text", () => {
    const raw = "Answer.|||SUGGEST: What GPA is required for grad school?";
    const result = parseAnswer(raw);
    expect(result.suggestions).toEqual(["What GPA is required for grad school?"]);
  });

  it("caps suggestions at 3", () => {
    const raw = "Answer.|||SUGGEST: Q1? | Q2? | Q3? | Q4?";
    const result = parseAnswer(raw);
    expect(result.suggestions).toHaveLength(3);
  });
});
