import torch
from torch import nn

from model.ccrr import CandidateContextEncoder, MaskedHybridPool, SCACRRModule


def test_hybrid_pool_output_shape():
    pool = MaskedHybridPool(
        channels=4,
        output_dim=6,
        topk_ratio=0.25,
        minimum_topk=2,
    )
    feature = torch.randn(3, 4, 5, 7)
    mask = torch.ones(3, 1, 5, 7)

    output = pool(feature, mask)

    assert output.shape == (3, 6)


def test_hybrid_pool_ignores_masked_values():
    pool = MaskedHybridPool(channels=2, output_dim=6, topk_ratio=0.5)
    # Inspect the pooled statistics directly, without a learned projection.
    pool.projection = nn.Identity()
    mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    mask[0, 0, 1, 1] = True
    clean = torch.zeros(1, 2, 3, 3)
    clean[0, :, 0, 0] = torch.tensor([1.0, 3.0])
    clean[0, :, 1, 1] = torch.tensor([5.0, 7.0])
    poisoned = clean.clone()
    poisoned.masked_fill_(~mask.expand_as(poisoned), float("nan"))

    clean_output = pool(clean, mask)
    poisoned_output = pool(poisoned, mask)

    torch.testing.assert_close(poisoned_output, clean_output)


def test_hybrid_pool_single_peak_is_visible_to_max_and_topk():
    pool = MaskedHybridPool(channels=4, output_dim=12, topk_ratio=0.125)
    pool.projection = nn.Identity()
    feature = torch.zeros(1, 4, 7, 7)
    feature[:, :, 3, 3] = 10.0
    mask = torch.ones(1, 1, 7, 7)

    statistics = pool(feature, mask)
    average, maximum, topk_mean = statistics.chunk(3, dim=1)

    torch.testing.assert_close(average, feature.mean(dim=(2, 3)))
    assert torch.all(maximum > average)
    assert torch.all(topk_mean > average)


def test_hybrid_pool_empty_mask_is_finite():
    pool = MaskedHybridPool(
        channels=3,
        output_dim=5,
        topk_ratio=0.25,
        minimum_topk=64,
    )
    feature = torch.randn(2, 3, 4, 4)
    mask = torch.zeros(2, 1, 4, 4)

    output = pool(feature, mask)

    assert output.shape == (2, 5)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_hybrid_pool_has_gradient():
    torch.manual_seed(17)
    pool = MaskedHybridPool(channels=3, output_dim=5, topk_ratio=0.25)
    with torch.no_grad():
        pool.projection[0].weight.normal_(mean=0.0, std=0.2)
    feature = torch.randn(2, 3, 5, 5, requires_grad=True)
    mask = torch.zeros(2, 1, 5, 5, dtype=torch.bool)
    mask[0, :, 1:5, 1:4] = True
    mask[1, :, :4, 2:5] = True

    pool(feature, mask).square().mean().backward()

    assert feature.grad is not None
    assert feature.grad.abs().sum() > 0
    assert pool.projection[0].weight.grad is not None
    assert pool.projection[0].weight.grad.abs().sum() > 0


def test_avg_mode_preserves_legacy_output_and_state_dict_contract():
    torch.manual_seed(23)
    legacy_default = CandidateContextEncoder(
        feature_channels=2,
        num_scales=2,
        roi_size=5,
        hidden_dim=4,
        context_scale=2.0,
    )
    explicit_avg = CandidateContextEncoder(
        feature_channels=2,
        num_scales=2,
        roi_size=5,
        hidden_dim=4,
        context_scale=2.0,
        pooling_mode="avg",
    )
    explicit_avg.load_state_dict(legacy_default.state_dict(), strict=True)

    expected_keys = {
        "roi_encoder.0.weight",
        "roi_encoder.0.bias",
        "roi_encoder.1.weight",
        "roi_encoder.1.bias",
        "roi_encoder.3.weight",
        "roi_encoder.3.bias",
        "roi_encoder.4.weight",
        "roi_encoder.4.bias",
    }
    assert set(legacy_default.state_dict()) == expected_keys
    assert set(explicit_avg.state_dict()) == expected_keys
    assert explicit_avg.spatial_encoder is None
    assert explicit_avg.hybrid_pool is None

    feature_map = torch.randn(1, 2, 8, 8)
    boxes = torch.tensor([[0, 2, 2, 6, 6]], dtype=torch.float32)
    masks = torch.zeros(1, 8, 8, dtype=torch.bool)
    masks[0, 2:6, 2:6] = True
    scale_features = torch.randn(1, 3)
    default_output = legacy_default(
        feature_map,
        boxes,
        scale_features=scale_features,
        candidate_masks=masks,
    )
    explicit_output = explicit_avg(
        feature_map,
        boxes,
        scale_features=scale_features,
        candidate_masks=masks,
    )

    torch.testing.assert_close(explicit_output, default_output, atol=0.0, rtol=0.0)


def test_candidate_context_encoder_hybrid_shape_and_gradient():
    encoder = CandidateContextEncoder(
        feature_channels=2,
        num_scales=2,
        roi_size=5,
        hidden_dim=4,
        context_scale=2.0,
        pooling_mode="avg_max_topk",
        topk_ratio=0.25,
        minimum_topk=2,
    )
    feature_map = torch.randn(1, 2, 8, 8, requires_grad=True)
    boxes = torch.tensor([[0, 2, 2, 6, 6]], dtype=torch.float32)
    masks = torch.zeros(1, 8, 8, dtype=torch.bool)
    masks[0, 2:6, 2:6] = True
    scale_features = torch.randn(1, 3)

    relation = encoder(
        feature_map,
        boxes,
        scale_features=scale_features,
        candidate_masks=masks,
    )

    assert relation.shape == (1, 4 * 4 + 3)
    assert encoder.core_encoder is encoder.context_encoder
    assert encoder.core_encoder is encoder.spatial_encoder
    relation.square().mean().backward()
    assert feature_map.grad is not None
    assert feature_map.grad.abs().sum() > 0


def test_sca_module_threads_hybrid_pool_parameters():
    module = SCACRRModule(
        feature_channels=4,
        num_scales=2,
        hidden_dim=8,
        pooling_mode="avg_max_topk",
        topk_ratio=0.2,
        minimum_topk=3,
    )

    assert module.encoder.pooling_mode == "avg_max_topk"
    assert module.encoder.hybrid_pool is not None
    assert module.encoder.hybrid_pool.topk_ratio == 0.2
    assert module.encoder.hybrid_pool.minimum_topk == 3
