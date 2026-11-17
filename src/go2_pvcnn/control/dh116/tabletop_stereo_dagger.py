"""Online Student-visited DAgger; Teacher queries never advance the world."""
from .tabletop_stereo_teacher_collector import collect_step as _collect_step


SCHEDULES=dict(D0=1.,D1=.5,D2=.2,D3=.1,D4=0.)


def collect_step(env,student,teacher,*,teacher_probability=0.,**options):
    if student is None:raise ValueError('DAgger requires a sensor-only Student policy')
    return _collect_step(env,student,teacher,teacher_probability=teacher_probability,**options)


def run(env,student,teacher,steps,*,stage='D4',teacher_probability=None,**options):
    if stage not in SCHEDULES:raise ValueError('explicit D0..D4 ownership schedule required')
    if type(steps) is not int or steps<1:raise ValueError('positive bounded online collection steps required')
    probability=SCHEDULES[stage] if teacher_probability is None else teacher_probability
    rows=[collect_step(env,student,teacher,teacher_probability=probability,**options) for _ in range(steps)]
    return dict(records=rows,stage=stage,mode='teacher_reference' if probability==1 else 'online_dagger',
        teacher_query_count=sum(row['teacher_query_count'] for row in rows),
        student_owned_samples=sum(row['dagger_student_samples'] for row in rows),
        native_qualified=False,schema_only=bool(options.get('schema_only',False)))
