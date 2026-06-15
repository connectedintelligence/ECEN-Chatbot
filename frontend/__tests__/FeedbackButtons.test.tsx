import { render, screen, fireEvent } from "@testing-library/react";
import { FeedbackButtons } from "../components/FeedbackButtons";

describe("FeedbackButtons (issue #15: feedback must be changeable)", () => {
  it("reports the clicked rating when none is set yet", () => {
    const onRate = jest.fn();
    render(<FeedbackButtons onRate={onRate} />);
    fireEvent.click(screen.getByLabelText("Good answer"));
    expect(onRate).toHaveBeenCalledWith("up");
  });

  it("after liking, the dislike button stays clickable so the vote can switch", () => {
    const onRate = jest.fn();
    render(<FeedbackButtons feedback="up" onRate={onRate} />);

    // The selected thumb is disabled; the opposite thumb is NOT.
    expect(screen.getByLabelText("Good answer")).toBeDisabled();
    expect(screen.getByLabelText("Bad answer")).not.toBeDisabled();

    fireEvent.click(screen.getByLabelText("Bad answer"));
    expect(onRate).toHaveBeenCalledWith("down");
  });

  it("after disliking, the like button stays clickable (symmetric)", () => {
    const onRate = jest.fn();
    render(<FeedbackButtons feedback="down" onRate={onRate} />);
    expect(screen.getByLabelText("Bad answer")).toBeDisabled();
    expect(screen.getByLabelText("Good answer")).not.toBeDisabled();
  });
});
