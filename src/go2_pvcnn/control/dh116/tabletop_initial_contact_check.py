"""Check measured reset geometry without pretending a pad is a contact."""
import torch
from .tabletop_policy_force_region import _radial_z

CONTRACT=dict(version='measured_initial_pad_below_axis_v1',
    maximum_pad_radial_z_m=-.0001,minimum_clearance_m=.0003,
    reference='four_thumb_index_pad_points_from_simulator_FK',
    contacts='separate_actual_sensor_reports_not_synthetic_pad_contacts')


def validate_initial_contact(measurement):
    radial=_radial_z(measurement['pad_relative'],measurement['object_quat'])
    clearance=measurement['clearance']
    if not torch.isfinite(radial).all() or not (radial<-.0001).all():
        raise ValueError('initial measured pad references are not all below the object axis')
    if not torch.isfinite(clearance).all() or not (clearance>=.0003).all():
        raise ValueError('initial measured table clearance is below 0.3mm')
    contacts=measurement['contact_valid']
    return dict(CONTRACT,all_pad_references_below_axis=True,
        pad_radial_z_m=radial.detach().cpu().tolist(),minimum_measured_clearance_m=float(clearance.min()),
        measured_contact_hands=int(contacts.sum()),
        actual_contact_below_axis_hands=int((contacts&measurement.get('contact_all_below_centerline',torch.zeros_like(contacts))).sum()),
        grasp_qualified=False)
