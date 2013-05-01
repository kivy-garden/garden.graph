'''
Graph
======

The :class:`Graph` widget is widget for displaying plots.

'''

__all__ = ('Graph', 'MeshLinePlot')

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.properties import NumericProperty, BooleanProperty,\
    BoundedNumericProperty, StringProperty, ListProperty, ObjectProperty,\
    DictProperty, AliasProperty
from kivy.clock import Clock
from kivy.graphics import Mesh, Color, Rotate, StencilUse
from kivy.graphics.transformation import Matrix
from kivy import metrics
from math import log10, floor, ceil
from decimal import Decimal


class Graph(Widget):
    '''Graph class, see module documentation for more information.
    '''

    # triggers a full reload of graphics
    _trigger = ObjectProperty(None)
    # triggers only a repositioning of objects due to size/pos updates
    _trigger_size = ObjectProperty(None)
    # holds widget with the x-axis label
    _xlabel = ObjectProperty(None)
    # holds widget with the y-axis label
    _ylabel = ObjectProperty(None)
    # holds all the x-axis tick mark labels
    _x_grid_label = ListProperty([])
    # holds all the y-axis tick mark labels
    _y_grid_label = ListProperty([])
    # the mesh drawing all the ticks/grids
    _mesh = ObjectProperty(None)
    # the mesh which draws the surrounding rectangle
    _mesh_rect = ObjectProperty(None)
    # a list of locations of major and minor ticks. The values are not
    # but is in the axis min - max range
    _ticks_majorx = ListProperty([])
    _ticks_minorx = ListProperty([])
    _ticks_majory = ListProperty([])
    _ticks_minory = ListProperty([])

    def __init__(self, **kwargs):
        super(Graph, self).__init__(**kwargs)

        self._mesh = Mesh(mode='lines')
        self._mesh_rect = Mesh(mode='line_strip')
        val = 0.25
        self.canvas.add(Color(1 * val, 1 * val, 1 * val))
        self.canvas.add(self._mesh)
        self.canvas.add(Color(1, 1, 1))
        self.canvas.add(self._mesh_rect)
        mesh = self._mesh_rect
        mesh.vertices = [0] * (5 * 4)
        mesh.indices = [k for k in xrange(5)]

        self._trigger = Clock.create_trigger(self._redraw_all)
        self._trigger_size = Clock.create_trigger(self._redraw_size)

        self.bind(center=self._trigger_size, padding=self._trigger_size,
                  font_size=self._trigger_size, plots=self._trigger_size,
                  x_grid=self._trigger_size, y_grid=self._trigger_size,
                  draw_border=self._trigger_size)
        self.bind(xmin=self._trigger, xmax=self._trigger,
                  xlog=self._trigger, x_ticks_major=self._trigger,
                  x_ticks_minor=self._trigger,
                  xlabel=self._trigger, x_grid_label=self._trigger,
                  ymin=self._trigger, ymax=self._trigger,
                  ylog=self._trigger, y_ticks_major=self._trigger,
                  y_ticks_minor=self._trigger,
                  ylabel=self._trigger, y_grid_label=self._trigger)
        self._trigger()

    def _get_ticks(self, major, minor, log, s_min, s_max):
        if major and s_max > s_min:
            if log:
                s_min = log10(s_min)
                s_max = log10(s_max)
                # count the decades in min - max. This is in actual decades,
                # not logs.
                n_decades = floor(s_max - s_min)
                # for the fractional part of the last decade, we need to
                # convert the log value, x, to 10**x but need to handle
                # differently if the last incomplete decade has a decade
                # boundary in it
                if floor(s_min + n_decades) != floor(s_max):
                    n_decades += 1 - (10 ** (s_min + n_decades + 1) - 10 **
                                      s_max) / 10 ** floor(s_max + 1)
                else:
                    n_decades += ((10 ** s_max - 10 ** (s_min + n_decades)) /
                                  10 ** floor(s_max + 1))
                # this might be larger than what is needed, but we delete
                # excess later
                n_ticks_major = n_decades / float(major)
                n_ticks = int(floor(n_ticks_major * (minor if minor >=
                                                     1. else 1.0))) + 2
                # in decade multiples, e.g. 0.1 of the decade, the distance
                # between ticks
                decade_dist = major / float(minor if minor else 1.0)

                points_minor = [0] * n_ticks
                points_major = [0] * n_ticks
                k = 0  # position in points major
                k2 = 0  # position in points minor
                # because each decade is missing 0.1 of the decade, if a tick
                # falls in < min_pos skip it
                min_pos = 0.1 - 0.00001 * decade_dist
                s_min_low = floor(s_min)
                # first real tick location. value is in fractions of decades
                # from the start we have to use decimals here, otherwise
                # floating point inaccuracies results in bad values
                start_dec = ceil((10 ** Decimal(s_min - s_min_low - 1)) /
                                 Decimal(decade_dist)) * decade_dist
                count_min = (0 if not minor else
                             floor(start_dec / decade_dist) % minor)
                start_dec += s_min_low
                count = 0  # number of ticks we currently have passed start
                while True:
                    # this is the current position in decade that we are.
                    # e.g. -0.9 means that we're at 0.1 of the 10**ceil(-0.9)
                    # decade
                    pos_dec = start_dec + decade_dist * count
                    pos_dec_low = floor(pos_dec)
                    diff = pos_dec - pos_dec_low
                    zero = abs(diff) < 0.001 * decade_dist
                    if zero:
                        # the same value as pos_dec but in log scale
                        pos_log = pos_dec_low
                    else:
                        pos_log = log10((pos_dec - pos_dec_low
                                         ) * 10 ** ceil(pos_dec))
                    if pos_log > s_max:
                        break
                    count += 1
                    if zero or diff >= min_pos:
                        if minor and not count_min % minor:
                            points_major[k] = pos_log
                            k += 1
                        else:
                            points_minor[k2] = pos_log
                            k2 += 1
                    count_min += 1
                #n_ticks = len(points)
            else:
                # distance between each tick
                tick_dist = major / float(minor if minor else 1.0)
                n_ticks = int(floor((s_max - s_min) / tick_dist) + 1)
                points_major = [0] * int(floor((s_max - s_min) / float(major))
                                         + 1)
                points_minor = [0] * (n_ticks - len(points_major) + 1)
                k = 0  # position in points major
                k2 = 0  # position in points minor
                for m in xrange(0, n_ticks):
                    if minor and m % minor:
                        points_minor[k2] = m * tick_dist + s_min
                        k2 += 1
                    else:
                        points_major[k] = m * tick_dist + s_min
                        k += 1
            del points_major[k:]
            del points_minor[k2:]
        else:
            points_major = []
            points_minor = []
        return points_major, points_minor

    def _update_labels(self):
        xlabel = self._xlabel
        ylabel = self._ylabel
        x = self.x
        y = self.y
        width = self.width
        height = self.height
        padding = self.padding
        x_next = padding + x
        y_next = padding + y
        xextent = x + width
        yextent = y + height
        ymin = self.ymin
        ymax = self.ymax
        xmin = self.xmin
        # set up x and y axis labels
        if xlabel:
            xlabel.text = self.xlabel
            xlabel.texture_update()
            xlabel.size = xlabel.texture_size
            xlabel.pos = (x + width / 2. - xlabel.width / 2., padding + y)
            y_next += padding + xlabel.height
        if ylabel:
            ylabel.text = self.ylabel
            ylabel.texture_update()
            ylabel.size = ylabel.texture_size
            ylabel.pos = (padding + x, y + height / 2. - ylabel.height / 2.)
            x_next += padding + ylabel.width
        xpoints = self._ticks_majorx
        xlabels = self._x_grid_label
        xlabel_grid = self.x_grid_label
        ylabel_grid = self.y_grid_label
        ypoints = self._ticks_majory
        ylabels = self._y_grid_label
        # now x and y tick mark labels
        if len(ylabels) and ylabel_grid:
            # horizontal size of the largest tick label, to have enough room
            ylabels[0].text = str(ypoints[0])
            ylabels[0].texture_update()
            y1 = ylabels[0].texture_size
            ylabels[0].text = str(ypoints[-1])
            ylabels[0].texture_update()
            y2 = ylabels[0].texture_size
            y_start = y_next + (padding + y1[1] if len(xlabels) and xlabel_grid
                                else 0) +\
                                (padding + y1[1] if not y_next else 0)
            yextent = y + height - padding - y1[1] / 2.
            if self.ylog:
                ymax = log10(ymax)
                ymin = log10(ymin)
            ratio = (yextent - y_start) / float(ymax - ymin)
            y_start -= y1[1] / 2.
            func = (lambda x: 10 ** x) if self.ylog else lambda x: x
            for k in xrange(len(ylabels)):
                ylabels[k].text = '%g' % func(ypoints[k])
                ylabels[k].size = ylabels[k].texture_size
                ylabels[k].pos = (x_next, y_start + (ypoints[k] - ymin) *
                                  ratio)
            x_next += max(y1[0], y2[0]) + padding
        if len(xlabels) and xlabel_grid:
            func = log10 if self.xlog else lambda x: x
            # find the distance from the end that'll fit the last tick label
            xlabels[0].text = str(func(xpoints[-1]))
            xlabels[0].texture_update()
            xextent = x + width - xlabels[0].texture_size[0] / 2. - padding
            # find the distance from the start that'll fit the first tick label
            if not x_next:
                xlabels[0].text = str(func(xpoints[0]))
                xlabels[0].texture_update()
                x_next = padding + xlabels[0].texture_size[0] / 2.
            xmin = func(xmin)
            ratio = (xextent - x_next) / float(func(self.xmax) - xmin)
            func = (lambda x: 10 ** x) if self.xlog else lambda x: x
            for k in xrange(len(xlabels)):
                xlabels[k].text = str(func(xpoints[k]))
                # update the size so we can center the labels on ticks
                xlabels[k].texture_update()
                xlabels[k].size = xlabels[k].texture_size
                xlabels[k].pos = (x_next + (xpoints[k] - xmin) * ratio -
                                  xlabels[k].texture_size[0] / 2., y_next)
            y_next += padding + xlabels[0].texture_size[1]
        # now re-center the x and y axis labels
        if xlabel:
            xlabel.x = x_next + (xextent - x_next) / 2. - xlabel.width / 2.
        if ylabel:
            ylabel.y = y_next + (yextent - y_next) / 2. - ylabel.height / 2.
        return x_next, y_next, xextent, yextent

    def _update_ticks(self, size):
        # re-compute the positions of the bounding rectangle
        mesh = self._mesh_rect
        vert = mesh.vertices
        if self.draw_border:
            vert[0] = size[0]
            vert[1] = size[1]
            vert[4] = size[2]
            vert[5] = size[1]
            vert[8] = size[2]
            vert[9] = size[3]
            vert[12] = size[0]
            vert[13] = size[3]
            vert[16] = size[0]
            vert[17] = size[1]
        else:
            vert[0:18] = [0 for k in xrange(18)]
        mesh.vertices = vert
        # re-compute the positions of the x/y axis ticks
        mesh = self._mesh
        vert = mesh.vertices
        start = 0
        xpoints = self._ticks_majorx
        ypoints = self._ticks_majory
        ylog = self.ylog
        xlog = self.xlog
        xmin = self.xmin
        xmax = self.xmax
        if xlog:
            xmin = log10(xmin)
            xmax = log10(xmax)
        ymin = self.ymin
        ymax = self.ymax
        if ylog:
            xmin = log10(ymin)
            ymax = log10(ymax)
        if len(xpoints):
            top = size[3] if self.x_grid else metrics.dp(12) + size[1]
            ratio = (size[2] - size[0]) / float(xmax - xmin)
            for k in xrange(start, len(xpoints) + start):
                vert[k * 8] = size[0] + (xpoints[k - start] - xmin) * ratio
                vert[k * 8 + 1] = size[1]
                vert[k * 8 + 4] = vert[k * 8]
                vert[k * 8 + 5] = top
            start += len(xpoints)
        if len(ypoints):
            top = size[2] if self.y_grid else metrics.dp(12) + size[0]
            ratio = (size[3] - size[1]) / float(ymax - ymin)
            for k in xrange(start, len(ypoints) + start):
                vert[k * 8 + 1] = size[1] + (ypoints[k - start] - ymin) * ratio
                vert[k * 8 + 5] = vert[k * 8 + 1]
                vert[k * 8] = size[0]
                vert[k * 8 + 4] = top
        mesh.vertices = vert

    def _update_plots(self, size):
        ylog = self.ylog
        xlog = self.xlog
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        for plot in self.plots:
            plot._update(xlog, xmin, xmax, ylog, ymin, ymax, size)

    def _redraw_all(self, *args):
        # add/remove all the required labels
        font_size = self.font_size
        if self.xlabel:
            if not self._xlabel:
                xlabel = Label(font_size=font_size)
                self.add_widget(xlabel)
                self._xlabel = xlabel
        else:
            xlabel = self._xlabel
            if xlabel:
                self.remove_widget(xlabel)
                self._xlabel = None
        grids = self._x_grid_label
        xpoints_major, xpoints_minor = self._get_ticks(self.x_ticks_major,
                                                       self.x_ticks_minor,
                                                       self.xlog, self.xmin,
                                                       self.xmax)
        self._ticks_majorx = xpoints_major
        self._ticks_minorx = xpoints_minor
        if not self.x_grid_label:
            n_labels = 0
        else:
            n_labels = len(xpoints_major)
        for k in xrange(n_labels, len(grids)):
            self.remove_widget(grids[k])
        del grids[n_labels:]
        grid_len = len(grids)
        grids.extend([None] * (n_labels - len(grids)))
        for k in xrange(grid_len, n_labels):
            grids[k] = Label(font_size=font_size)
            self.add_widget(grids[k])

        if self.ylabel:
            if not self._ylabel:
                ylabel = Label(font_size=font_size)
                self.add_widget(ylabel)
                rot = Rotate()
                rot.set(90, 0, 0, 1)
                #ylabel.canvas.before.add(rot)
                self._ylabel = ylabel
        else:
            ylabel = self._ylabel
            if ylabel:
                self.remove_widget(ylabel)
                self._ylabel = None
        grids = self._y_grid_label
        ypoints_major, ypoints_minor = self._get_ticks(self.y_ticks_major,
                                                       self.y_ticks_minor,
                                                       self.ylog, self.ymin,
                                                       self.ymax)
        self._ticks_majory = ypoints_major
        self._ticks_minory = ypoints_minor
        if not self.y_grid_label:
            n_labels = 0
        else:
            n_labels = len(ypoints_major)
        for k in xrange(n_labels, len(grids)):
            self.remove_widget(grids[k])
        del grids[n_labels:]
        grid_len = len(grids)
        grids.extend([None] * (n_labels - len(grids)))
        for k in xrange(grid_len, n_labels):
            grids[k] = Label(font_size=font_size)
            self.add_widget(grids[k])

        mesh = self._mesh
        n_points = (len(xpoints_major) + len(xpoints_minor) +
                    len(ypoints_major) + len(ypoints_minor))
        mesh.vertices = [0] * (n_points * 8)
        mesh.indices = [k for k in xrange(n_points * 2)]
        self._redraw_size()

    def _redraw_size(self, *args):
        # size a 4-tuple describing the bounding box in which we can draw
        # graphs, it's (x0, y0, x1, y1), which correspond with the bottom left
        # and top right corner locations, respectively
        size = self._update_labels()
        self._update_ticks(size)
        self._update_plots(size)

    def add_plot(self, plot):
        for group in plot._drawing():
            self.canvas.add(group)
        self.plots = self.plots + [plot]

    def remove_plot(self, plot):
        self.canvas.remove_group(plot._get_group())
        self.plots.remove(plot)

    xmin = NumericProperty(0.)
    '''Minimum value allowed for :data:`value`.

    :data:`min` is a :class:`~kivy.properties.NumericProperty`, default to 0.
    '''

    xmax = NumericProperty(100.)
    '''Maximum value allowed for :data:`value`.

    :data:`max` is a :class:`~kivy.properties.NumericProperty`, default to 100.
    '''

    xlog = BooleanProperty(False)
    '''Determines whether the ticks should be displayed logarithmically (True)
    or linearly (False).

    :data:`log` is a :class:`~kivy.properties.BooleanProperty`, defaults
    to False.
    '''

    x_ticks_major = BoundedNumericProperty(0, min=0)
    '''Distance between major tick marks.

    Determines the distance between the major tick marks. Major tick marks
    start from min and re-occur at every ticks_major until :data:`max`.
    If :data:`max` doesn't overlap with a integer multiple of ticks_major,
    no tick will occur at :data:`max`. Zero indicates no tick marks.

    If :data:`log` is true, then this indicates the distance between ticks
    in multiples of current decade. E.g. if :data:`min_log` is 0.1 and
    ticks_major is 0.1, it means there will be a tick at every 10th of the
    decade, i.e. 0.1 ... 0.9, 1, 2... If it is 0.3, the ticks will occur at
    0.1, 0.3, 0.6, 0.9, 2, 5, 8, 10. You'll notice that it went from 8 to 10
    instead of to 20, that's so that we can say 0.5 and have ticks at every
    half decade, e.g. 0.1, 0.5, 1, 5, 10, 50... Similarly, if ticks_major is
    1.5, there will be ticks at 0.1, 5, 100, 5,000... Also notice, that there's
    always a major tick at the start. Finally, if e.g. :data:`min_log` is 0.6
    and this 0.5 there will be ticks at 0.6, 1, 5...

    :data:`ticks_major` is a :class:`~kivy.properties.BoundedNumericProperty`,
    defaults to 0.
    '''

    x_ticks_minor = BoundedNumericProperty(0, min=0)
    '''The number of sub-intervals that divide ticks_major.

    Determines the number of sub-intervals into which ticks_major is divided,
    if non-zero. The actual number of minor ticks between the major ticks is
    ticks_minor - 1. Only used if ticks_major is non-zero. If there's no major
    tick at max then the number of minor ticks after the last major
    tick will be however many ticks fit until max.

    If self.log is true, then this indicates the number of intervals the
    distance between major ticks is divided. The result is the number of
    multiples of decades between ticks. I.e. if ticks_minor is 10, then if
    ticks_major is 1, there will be ticks at 0.1, 0.2...0.9, 1, 2, 3... If
    ticks_major is 0.3, ticks will occur at 0.1, 0.12, 0.15, 0.18... Finally,
    as is common, if ticks major is 1, and ticks minor is 5, there will be
    ticks at 0.1, 0.2, 0.4... 0.8, 1, 2...

    :data:`ticks_minor` is a :class:`~kivy.properties.BoundedNumericProperty`,
    defaults to 0.
    '''

    x_grid = BooleanProperty(False)

    x_grid_label = BooleanProperty(False)

    xlabel = StringProperty('')

    ymin = NumericProperty(0.)
    '''Minimum value allowed for :data:`value`.

    :data:`min` is a :class:`~kivy.properties.NumericProperty`, default to 0.
    '''

    ymax = NumericProperty(100.)
    '''Maximum value allowed for :data:`value`.

    :data:`max` is a :class:`~kivy.properties.NumericProperty`, default to 100.
    '''

    ylog = BooleanProperty(False)
    '''Determines whether the ticks should be displayed logarithmically (True)
    or linearly (False).

    :data:`log` is a :class:`~kivy.properties.BooleanProperty`, defaults
    to False.
    '''

    y_ticks_major = BoundedNumericProperty(0, min=0)
    '''Distance between major tick marks. See :data:`x_ticks_major`.

    :data:`ticks_major` is a :class:`~kivy.properties.BoundedNumericProperty`,
    defaults to 0.
    '''

    y_ticks_minor = BoundedNumericProperty(0, min=0)
    '''The number of sub-intervals that divide ticks_major.
    See :data:`x_ticks_minor`.

    :data:`ticks_minor` is a :class:`~kivy.properties.BoundedNumericProperty`,
    defaults to 0.
    '''

    y_grid = BooleanProperty(False)

    y_grid_label = BooleanProperty(False)

    ylabel = StringProperty('')

    padding = NumericProperty(metrics.dp(5))
    '''Padding of the slider. The padding is used for graphical representation
    and interaction. It prevents the cursor from going out of the bounds of the
    slider bounding box.

    By default, padding is 10. The range of the slider is reduced from padding
    2 on the screen. It allows drawing a cursor of 20px width, without having
    the cursor going out of the widget.

    :data:`padding` is a :class:`~kivy.properties.NumericProperty`, default to
    10.
    '''

    font_size = NumericProperty('15sp')
    '''Font size of the labels.

    :data:`font_size` is a :class:`~kivy.properties.NumericProperty`, defaults
    to 15sp.
    '''

    draw_border = BooleanProperty(True)

    plots = ListProperty([])


class MeshLinePlot(Widget):

    _mesh = ObjectProperty(None)
    _color = ObjectProperty(None)
    _trigger = ObjectProperty(None)
    # most recent values of the params
    _params = DictProperty({'xlog': False, 'xmin': 0, 'xmax': 100,
                            'ylog': False, 'ymin': 0, 'ymax': 100,
                            'size': (0, 0, 0, 0)})

    def __init__(self, **kwargs):
        self._mesh = Mesh(group='LinePlot%d' % id(self))
        self._color = Color(1, 1, 1, group='LinePlot%d' % id(self))
        super(MeshLinePlot, self).__init__(**kwargs)

        self._trigger = Clock.create_trigger(self._redraw)
        self.bind(_params=self._trigger, points=self._trigger)

    def _update(self, xlog, xmin, xmax, ylog, ymin, ymax, size):
        self._params = {'xlog': xlog, 'xmin': xmin, 'xmax': xmax, 'ylog': ylog,
                        'ymin': ymin, 'ymax': ymax, 'size': size}

    def _redraw(self, *args):
        points = self.points
        mesh = self._mesh
        vert = mesh.vertices
        ind = mesh.indices
        params = self._params
        funcx = log10 if params['xlog'] else lambda x: x
        funcy = log10 if params['ylog'] else lambda x: x
        xmin = funcx(params['xmin'])
        ymin = funcy(params['ymin'])
        diff = len(points) - len(vert) / 4
        size = params['size']
        ratiox = (size[2] - size[0]) / float(funcx(params['xmax']) - xmin)
        ratioy = (size[3] - size[1]) / float(funcy(params['ymax']) - ymin)
        if diff < 0:
            del vert[4 * len(points):]
            del ind[len(points):]
        elif diff > 0:
            ind.extend(xrange(len(ind), len(ind) + diff))
            vert.extend([0] * (diff * 4))
        for k in xrange(len(points)):
            vert[k * 4] = (funcx(points[k][0]) - xmin) * ratiox + size[0]
            vert[k * 4 + 1] = (funcy(points[k][1]) - ymin) * ratioy + size[1]
        mesh.vertices = vert

    def _get_group(self):
        return 'LinePlot%d' % id(self)

    def _drawing(self):
        return [self._color, self._mesh]

    def _set_mode(self, value):
        self._mesh.mode = value
        print value
    mode = AliasProperty(lambda self: self._mesh.mode, _set_mode)
    '''Minimum value allowed for :data:`value_log` when using logarithms.

    :data:`min_log` is a :class:`~kivy.properties.AliasProperty`
    of :data:`min`.
    '''

    def _set_color(self, value):
        self._color.rgba = value
    color = AliasProperty(lambda self: self._color.rgba, _set_color)
    '''Minimum value allowed for :data:`value_log` when using logarithms.

    :data:`min_log` is a :class:`~kivy.properties.AliasProperty`
    of :data:`min`.
    '''

    points = ListProperty([])
    '''Minimum value allowed for :data:`value_log` when using logarithms.

    :data:`min_log` is a :class:`~kivy.properties.AliasProperty`
    of :data:`min`.
    '''


if __name__ == '__main__':
    from math import sin, cos

    class TestApp(App):

        def build(self):

            graph = Graph(xlabel='Cheese', ylabel='Apples', x_ticks_minor=5,
                          x_ticks_major=25, y_ticks_minor=5, y_ticks_major=1,
                          y_grid_label=True, x_grid_label=True, padding=5,
                          xlog=False, ylog=False, x_grid=True, y_grid=True,
                          xmin=-50, xmax=50, ymin=-1, ymax=1)
            plot = MeshLinePlot(mode='line_strip', color=[1, 0, 0, 1])
            plot.points = [(x / 10., sin(x / 50.)) for x in xrange(-500, 501)]
            graph.add_plot(plot)
            plot = MeshLinePlot(mode='line_strip', color=[0, 1, 0, 1])
            plot.points = [(x / 10., cos(x / 50.)) for x in xrange(-500, 501)]
            graph.add_plot(plot)
            plot = MeshLinePlot(mode='line_strip', color=[0, 0, 1, 1])
            graph.add_plot(plot)
            plot.points = [(x, x / 50.) for x in xrange(-50, 51)]
            return graph

    TestApp().run()
