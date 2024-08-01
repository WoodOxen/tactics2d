import numpy as np
import math

class PredictionLoader:
    """
    A PredictionLoader class that serves as a utility for loading prediction data and processing it according to the given scenario and agents.

    This class can be instantiated and called with data and prediction results to output a structured format containing predictions for each agent.
    
    Examples of how to use PredictionLoader in practice:
    
    >>> from prediction.predictor import PredictionLoader
    >>> predictor = PredictionLoader()
    >>> from DataLoader import WaymoDL
    >>> loader = WaymoDL('./waymo_tf_example/training')
    >>> data, _ = loader.get_next()
    >>> output = predictor(data)
    >>> print(output)
    """

    def __init__(self, all_agents=False):
        """
        This function initializes a new instance of the PredictionLoader class with an option to include all agents.
        
        Args:
            all_agents (bool): A flag that indicates whether all agents should be processed for predictions. Defaults to False.
        """
        self.all_agents = all_agents

    def __call__(self, data, prediction_loaded, file_mode=0):
        """
        This function processes the provided scenario data and attempt to load associated prediction results for each agent in the scenario.
        
        Args:
            data (dict): Information about the scenario being loaded which includes agent details such as their poses, speeds, and shapes.
            prediction_loaded (dict): A preloaded dictionary containing the results for predictions that are to be loaded into this scenario.
            file_mode (int): Mode specifying the structure of the prediction loaded dictionary; mode 0 corresponds to a specific expected structure.
        
        Returns:
            dict: A dictionary mapping agent identifiers to their respective predictions. Each prediction is another dictionary with 'pred_trajectory' and 'pred_yaw', and optionally 'pred_scores'.
        """
        # Initialize the scenario and containers for predictions and agents
        scenario = data['scenario']
        agent_to_pred = {}

        # Iterate over each agent in the scenario
        for agent_id in data['agent']:
            agent = data['agent'][agent_id]
            pose = agent['pose']
            speed = agent['speed']
            shape = agent['shape']
            type = int(agent['type'])
            to_predict = int(agent['to_predict'])

            # Process the agent if all agents need to be processed or if this specific agent needs a prediction
            if self.all_agents or to_predict:
                # Initialize arrays for predictions
                pred_trajectory = np.zeros([6, 80, 2])
                x, y = pose[-1, 0], pose[-1, 1]  # Assuming the last entry in the pose is the current position
                pred_yaw = np.zeros([6, 80])
                pred_scores = np.zeros(6)

                # Function to calculate the angle from the delta x and y
                def get_angle(x, y):
                    return math.atan2(y, x)

                # Calculate average velocity vector from the poses
                delta_list = [(agent['pose'][i, 0] - agent['pose'][i - 1, 0], agent['pose'][i, 1] - agent['pose'][i - 1, 1]) for i in range(6, 10)]
                delta_x, delta_y = np.mean(delta_list, axis=0)

                # Check if prediction data is available for the scenario and agent
                if scenario in prediction_loaded and agent_id in prediction_loaded[scenario] and 'rst' in prediction_loaded[scenario][agent_id]:
                    # Load prediction score and result for this agent
                    pred_scores = np.exp(prediction_loaded[scenario][agent_id]['score'])
                    loaded_pred = prediction_loaded[scenario][agent_id]['rst']

                    # Iterate over each possible future prediction and process
                    for each_prediction in range(6):
                        pred_trajectory[each_prediction, :, :] = loaded_pred[each_prediction, :, :]
                        # Calculate yaw for each time step in prediction
                        for i in range(80):
                            # Placeholder for incomplete logic to calculate predicted yaw angle
                            pass

                # If no prediction data is found, an alternative handling is required
                else:
                    # Logic to handle missing prediction data
                    pass

                # Assign the calculated prediction to the agent
                agent_to_pred[agent_id] = {
                    'pred_trajectory': pred_trajectory,
                    'pred_yaw': pred_yaw,
                    'pred_scores': pred_scores / np.sum(pred_scores) if np.sum(pred_scores) > 0.01 else pred_scores
                }

        # Return the dictionary containing predictions for all agents in the scenario
        return agent_to_pred

