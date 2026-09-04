import { INSTANCE_WRITE_MODES, MessageBuilders } from './messageTypes';

describe('MessageBuilders.runInstance', () => {
  test('defaults interactive inference to patch mode', () => {
    const message = MessageBuilders.runInstance('instance-model');

    expect(message.data).toEqual({
      model_registry_key: 'instance-model',
      write_mode: INSTANCE_WRITE_MODES.PATCH,
    });
  });

  test('includes inputs in websocket payload when provided', () => {
    const message = MessageBuilders.runInstance(
      'instance-model',
      INSTANCE_WRITE_MODES.PATCH,
      { parameters: { threshold: 0.75 } }
    );

    expect(message.data).toEqual({
      model_registry_key: 'instance-model',
      write_mode: INSTANCE_WRITE_MODES.PATCH,
      inputs: { parameters: { threshold: 0.75 } },
    });
  });
});

describe('MessageBuilders.runSegmentation', () => {
  test('builds payload without inputs when none provided', () => {
    const message = MessageBuilders.runSegmentation('sam-model', {
      point_prompts: [{ x: 0.1, y: 0.2, label: true }],
    });

    expect(message.data).toEqual({
      model_identifier: 'sam-model',
      model_key: 'sam-model',
      prompts: {
        point_prompts: [{ x: 0.1, y: 0.2, label: true }],
        box_prompt: null,
        polygon_prompt: null,
        circle_prompt: null,
      },
    });
  });

  test('includes inputs in websocket payload when provided', () => {
    const message = MessageBuilders.runSegmentation(
      'sam-model',
      { point_prompts: [{ x: 0.1, y: 0.2, label: true }] },
      { parameters: { threshold: 0.9 } }
    );

    expect(message.data).toEqual({
      model_identifier: 'sam-model',
      model_key: 'sam-model',
      prompts: {
        point_prompts: [{ x: 0.1, y: 0.2, label: true }],
        box_prompt: null,
        polygon_prompt: null,
        circle_prompt: null,
      },
      inputs: {
        parameters: { threshold: 0.9 },
      },
    });
  });
});

describe('MessageBuilders.runSuggestion', () => {
  test('builds payload without inputs when none provided', () => {
    const message = MessageBuilders.runSuggestion([10, 11], 'sam3-sugg', 2);

    expect(message.data).toEqual({
      seed_contour_ids: [10, 11],
      model_key: 'sam3-sugg',
      label_id: 2,
    });
  });

  test('includes inputs in websocket payload when provided', () => {
    const message = MessageBuilders.runSuggestion([10, 11], 'sam3-sugg', 2, {
      parameters: { mask_threshold: 0.8 },
      conditioning: { count: 4 },
    });

    expect(message.data).toEqual({
      seed_contour_ids: [10, 11],
      model_key: 'sam3-sugg',
      label_id: 2,
      inputs: {
        parameters: { mask_threshold: 0.8 },
        conditioning: { count: 4 },
      },
    });
  });
});
