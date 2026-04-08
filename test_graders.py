from graders import EasyGrader, MediumGrader, HardGrader

def test():
    for grader_class in [EasyGrader, MediumGrader, HardGrader]:
        grader = grader_class()
        score = grader.grade()
        print(f"{grader_class.__name__} score: {score}")
        assert 0 < score < 1, f"Score {score} out of range!"

if __name__ == "__main__":
    test()
    print("All graders passed local validation.")