# SynLogic Training Guidance

## How to Use SynLogic in Verl Training

### SynLogic Data Structure Overview

Each sample in SynLogic contains the following components:

* **data_source**: Indicates the specific task type of this sample
* **prompt**: The prompt used for RL training
* **ability**: Indicates the specific ability being trained
* **reward_model**: Provides the answer and solution
* **extra_info**: Contains **game_data_str**, which includes the specific properties of this sample, such as constraints

### Integration with Verl

We utilize only the **game_data_str** from the extra_info field to calculate rewards. A reference implementation is provided in `reward_example.py`.

#### Implementation Steps:

1. **Define the reward format**: Set up extraction process and format reward design
2. **Use data_source for verifier selection**: Determine which verifier to use based on the data_source field
3. **Call the verify function**: Execute the verification process with:
   - `game_data`: Loaded from the game_data_str
   - `response`: The model's response to be verified
   - Returns the correctness of the response


For detailed implementation, please refer to the `reward_example.py` file in the repository.