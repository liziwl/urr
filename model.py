from wtforms import Form, FloatField, validators, SelectField
from math import pi
import functools
from os import listdir
from os.path import isfile, join


def check_T(form, field):
    """Form validation: failure if T > 30 periods."""
    w = form.w.data
    T = field.data
    period = 2*pi/w
    if T > 30*period:
        num_periods = int(round(T/period))
        raise validators.ValidationError(
            'Cannot plot as much as %d periods! T<%.2f' %
            (num_periods, 30*period))

def check_interval(form, field, min_value=None, max_value=None):
    """For validation: failure if value is outside an interval."""
    failure = False
    if min_value is not None:
        if field.data < min_value:
            failure = True
    if max_value is not None:
        if field.data > max_value:
            failure = True
    if failure:
        raise validators.ValidationError(
            '%s=%s not in [%s, %s]' %
            (field.name, field.data,
             '-infty' if min_value is None else str(min_value),
             'infty'  if max_value is None else str(max_value)))

def interval(min_value=None, max_value=None):
    """Flask-compatible interface to check_interval."""
    return functools.partial(
        check_interval, min_value=min_value, max_value=max_value)


def find_files(files_path):
    return [f for f in listdir(files_path) if isfile(join(files_path, f))]


def build_files_choices(files_path):
    return [(join(files_path, a_file), a_file) for a_file in find_files(files_path)]


class InputForm(Form):

    user_review_files = SelectField(label=u'User Reviews File', choices=build_files_choices("./user_reviews_files/"))

    #A = FloatField(
    #    label='amplitude (m)', default=1.0,
    #    validators=[validators.NumberRange(0, 1E+20)])
    #b = FloatField(
    #    label='damping factor (kg/s)', default=0,
    #    validators=[validators.InputRequired(), interval(0,None)])
    #w = FloatField(
    #    label='frequency (1/s)', default=2*pi,
    #    validators=[validators.InputRequired()])
    #T = FloatField(
    #    label='time interval (s)', default=18,
    #    validators=[check_T, validators.InputRequired()])
