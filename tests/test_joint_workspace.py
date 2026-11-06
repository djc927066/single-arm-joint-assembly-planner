"""The table must come from SEALP, and the fixed assembly must stay on it."""
from pathlib import Path
import numpy as np
import pytest
from jbls.workspace import load_workspace, bridge_site, DEFAULT_CONFIG


def test_original_single_table_dimensions_and_bridge_containment():
    workspace = load_workspace()
    assert DEFAULT_CONFIG == Path(__file__).resolve().parents[1] / 'sealp/config/sample_config.yaml'
    np.testing.assert_allclose(workspace.table_size, [.6,.6,.02])
    np.testing.assert_allclose(workspace.table_center, [.2335,0.,-.01])
    assert workspace.top_z == 0
    base = np.array([[-.17,-.17,0.],[.17,.17,.02]])
    site = bridge_site(workspace, base)
    np.testing.assert_allclose(site, [.3535,0.,0.], atol=1e-7)
    np.testing.assert_array_less(site[:2]+base[1,:2], workspace.table_center[:2]+workspace.table_size[:2]/2)
    with pytest.raises(ValueError, match='outside the table'):
        bridge_site(workspace, base, requested_x=.4)


def test_unknown_environment_is_not_silently_omitted(tmp_path):
    import yaml
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    cfg['environment']['obstacles'].append(dict(name='wall', type='box', extent=[1,1,1], pos=[0,0,0]))
    path = tmp_path/'workspace.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match='extra obstacles'):
        load_workspace(path)
