import pytest
from scripts.amend_external_lvh_timing import amended_rows


def test_timing_amendment_preserves_original_metadata_and_membership():
    old=[{'id':'synthetic','fps':17.355,'n_frames':198,'true':0}]
    receipt=[{'id':'synthetic','csv_field_fps':17.355,'container_fps':58.,'decoded_frames':198,'pts_delta_max_error':1e-14}]
    new=amended_rows(old,receipt)
    assert new[0]['fps']==58 and new[0]['reported_csv_fps']==17.355
    assert old[0]['fps']==17.355 and new[0]['true']==old[0]['true']


def test_incomplete_or_ambiguous_timing_qualification_is_rejected():
    old=[{'id':'synthetic','fps':20.,'n_frames':20}]
    with pytest.raises(ValueError):amended_rows(old,[])
    r=[{'id':'synthetic','csv_field_fps':20.,'container_fps':50.,'decoded_frames':20,'pts_delta_max_error':.01}]
    with pytest.raises(ValueError):amended_rows(old,r)
