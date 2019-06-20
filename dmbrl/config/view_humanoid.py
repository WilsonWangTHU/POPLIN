if __name__ == '__main__':
    '''
    from dm_control import suite
    from dm_control import viewer
    import numpy as np

    test_env = suite.load(domain_name="humanoid", task_name="stand")
    action_spec = test_env.action_spec()

    def initialize_episode(physics):
        with physics.reset_context():
            physics.data.qpos[:] = 0.0
            physics.data.qpos[2] = 1.33
            physics.data.qvel[:] = 0.0
        print(physics.head_height())
        print(physics.head_height())
        print(physics.head_height())
    test_env.task.initialize_episode = initialize_episode

    # Define a uniform random policy.
    def random_policy(time_step):
      del time_step  # Unused.
      return np.random.uniform(low=action_spec.minimum,
                               high=action_spec.maximum,
                               size=action_spec.shape)

    # Launch the viewer application.
    viewer.launch(test_env, policy=random_policy)
    '''
    from dm_control import suite
    import matplotlib.pyplot as plt
    import numpy as np

    max_frame = 90

    width = 480
    height = 480
    video = np.zeros((90, height, 2 * width, 3), dtype=np.uint8)

    # Load one task:
    env = suite.load(domain_name="humanoid", task_name="walk")

    # Step through an episode and print out reward, discount and observation.
    action_spec = env.action_spec()
    time_step = env.reset()

    with env.physics.reset_context():
        env.physics.data.qpos[:] = 0.0
        env.physics.data.qpos[2] = 1.33
        env.physics.data.qvel[:] = 0.0
    head_pos = []
    while not time_step.last():
      for i in range(max_frame):
        action = np.random.uniform(action_spec.minimum,
                                   action_spec.maximum,
                                   size=action_spec.shape)
        time_step = env.step(action)

        head_pos.append(env.physics.head_height())
        video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                              env.physics.render(height, width, camera_id=1)])
        # print(time_step.reward, time_step.discount, time_step.observation)
      for i in range(max_frame):
        print(head_pos[i])
        img = plt.imshow(video[i])
        plt.pause(1)  # Need min display time > 0.0.
        plt.draw()
