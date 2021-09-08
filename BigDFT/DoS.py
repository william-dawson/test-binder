"""
This module is used to display the Density of States (DoS).

The Main Class of this module is :py:class:`DoS`. Essentially,
such class interpret the information coming from a set of energies and
represents in this way a curve that smears such energies.
"""


AU_eV = 27.21138386


class DiracSuperposition():
    """Sum of weighted Dirac deltas.

    Defines a superposition of Dirac deltas which can be used to
    plot the density of states. Such deltas are given on a per k-point
    basis and can be also equipped with a array of weights, which may be
    useful to define Partial Density of States.

    Args:
       dos (matrix): array containing the density of states per each k-point.
            Should be of shape 2, a list of energies per kpoint
       wgts(float, array, matrix): contains the weights of each of the
            k-points. Could be a constant, in which case the same weight is
            applied everywhere, a list (on a k-point basis),
            or a matrix, where each energy have its own weight.

    """
    def __init__(self, dos, wgts=[1.0]):
        import numpy as np
        self.dos = dos
        if isinstance(wgts, float):
            self.norm = [wgts]
        else:
            self.norm = wgts
        # set range for this distribution
        e_min = 1.e100
        e_max = -1.e100
        ddos = np.ravel(dos)
        if len(ddos) > 0:
            e_min = min(e_min, np.min(ddos) - 0.05 *
                        (np.max(ddos) - np.min(ddos)))
            e_max = max(e_max, np.max(ddos) + 0.05 *
                        (np.max(ddos) - np.min(ddos)))
        self.xlim = (e_min, e_max)

    def curve(self, xs, sigma, wgts=None):
        """Output the quantities to be plotted.

        Returns the x,y arrays that can be passed to plotting programs to
        represent the DoS.

        Args:
            xs (list): xrange of the output.
            sigma (float): smearing of the energies, as returned by
                :py:meth:`peaks`.
            wgts (list): array of further weights to be applied, if needed

        Returns:
            collections.NamedTuple: x,y numpy arrays of the curve in the
                specified range.
        """
        from numpy import ones
        from collections import namedtuple
        Curve = namedtuple('DoSCurve', 'x y')
        dos_g = 0.0
        idos = 0
        for norm, dos in zip(self.norm, self.dos):
            if wgts is not None:
                norms = wgts[idos]*norm
                idos += 1
            else:
                norms = ones(len(dos))*norm
            kptcurve = self.peaks(xs, dos, norms, sigma)
            dos_g += kptcurve
        return Curve(x=xs, y=dos_g)

    def peak(self, omega, e, sigma):
        """
        Define if a peak is a Gaussian or a Lorenzian (temporarily only the
        gaussian is defined)
        """
        import numpy as np
        nfac = np.sqrt(2.0*np.pi)
        val = np.exp(- (omega - e)**2 / (2.0 * sigma**2))/(nfac*sigma)
        return val

    def peaks(self, xs, dos, norms, sigma):
        """
        Return the array of the whole set of peaks
        """
        curve = 0.0
        for e, nrm in zip(dos, norms):
            curve += self.peak(xs, e, sigma)*nrm
        return curve


def _bandarray_to_data(jspin, bandarrays):
    lbl = 'up' if jspin == 0 else 'dw'
    kptlists = [[], []]
    for orbs in bandarrays:
        for ispin, norbs in enumerate(orbs.info):
            if norbs == 0 or ispin != jspin:
                continue
            # energy values
            kptlists[0].append(orbs[ispin, :norbs])
            # normalization
            kptlists[1].append(orbs.kwgt*(1.0-2*ispin))
            # print 'kpt',kptlists
    return kptlists, lbl if norbs != 0 else ''


class DoS():
    """Density of states class.

    Extract a quantity which is associated to the DoS, that can be plotted

    Args:
        bandarrays (list): a list of :py:class:`~BigDFT.BZ.BandArray`
            class instances.
        energies (array-like): a list of energies to be plotted.
            Useful for plotting a single DoS from an array.
        evals (dict): dictionary of eigenvalues from a logfile yaml
            serialization. May be useful if a
            :py:class:`~BigDFT.Logfiles.Logfile` class is not available.
        logfiles_dict(dict): dictionary of :py:class:`~BigDFT.Logfiles.Logfile`
            class instances as values to be employed to create
            the set of DoS. The keys of the dictionary are the labels of the
            corresponding curves.
        reference_fermi_level (float): useful in the case of a logfiles_dict.
            the value of the reference fermi level is compared with the
            fermi level of all the logfiles in order to align all the curves.
            It is assumed that the fermi levels are properly defined in all
            the logfiles.
        units (str): The units of the provided energies, and fermi_level.
            Should be `AU` of `eV`.
        label (str): label of the curve.
        sigma (float): smearing of the energies. Always provided in eV.
        fermi_level(float): set the value of the fermi energy.
        norm (float, list): weights of the :py:class:`DiracSuperposition`
           instantation for each of the k-points.
        sdos : spatial density of states instantiation
        **kwargs: keyword arguments of the :py:meth:`set_range` method.
    """

    def __init__(self, bandarrays=None, energies=None, evals=None,
                 logfiles_dict=None, units='eV', reference_fermi_level=None,
                 label='1', sigma=0.2, fermi_level=None, norm=1.0, sdos=None,
                 **kwargs):
        import numpy as np
        self.ens = []
        self.labels = []
        self.ef = None
        if bandarrays is not None:
            self.append_from_bandarray(bandarrays, label)
        if evals is not None:
            self.append_from_dict(evals, label)
        if energies is not None:
            self.append(np.array([energies]), label=label, units=units,
                        norm=(np.array([norm])
                        if isinstance(norm, float) else norm))
        if logfiles_dict is not None:
            shifts = {}
            for lb, log in logfiles_dict.items():
                self.append_from_bandarray(log.evals, label=lb)
                if reference_fermi_level is not None:
                    sh = reference_fermi_level - log.fermi_level
                    shifts[lb] = sh*self._conversion_factor(units='AU')
            if len(shifts) > 0:
                self.fermi_level(reference_fermi_level, units='AU')
                self.shift_curves(shifts)
        self.sigma = sigma
        self.fermi_level(fermi_level, units=units)
        self.set_range(**kwargs)
        if sdos is not None:
            self._embed_sdos(sdos)

    def _embed_sdos(self, sdos):
        self.sdos = []
        for i, xdos in enumerate(sdos):
            self.sdos.append({'coord': xdos['coord']})
            jdos = 0
            for subspin in xdos['dos']:
                if len(subspin[0]) == 0:
                    continue
                d = {'doslist': subspin}
                try:
                    self.ens[jdos]['sdos'].append(d)
                except KeyError:
                    self.ens[jdos]['sdos'] = [d]
                jdos += 1

    def append_from_bandarray(self, bandarrays, label):
        """Add a new band array to the previous DoS.

        This method can be called to include in the DoS the energies which
        come from another run.

        Args:
            bandarrays (BigDFT.BZ.BandArray): a instance of the eigenvalues
                which come from a logfile. Can be retrieved by the
                `BigDFT.Logfiles.Logfile.evals` attribute of the class.
            label (str): id of the run.
        """
        import numpy as np
        for jspin in range(2):
            kptlists, lbl = _bandarray_to_data(jspin, bandarrays)
            self.append(np.array(kptlists[0]), label=label+lbl, units='AU',
                        norm=np.array(kptlists[1]))

    def append_from_dict(self, evals, label):
        """Get the energies from the different flavours given by the dict."""
        import numpy as np
        evs = [[], []]
        ef = None
        for ev in evals:
            occ = self._get_ev(ev, ['e_occ', 'e_occupied'])
            if occ:
                ef = max(occ)
            vrt = self._get_ev(ev, ['e_vrt', 'e_virt'])
            eigen = False
            if occ:
                eigen = occ
            if vrt:
                eigen = vrt
            if not eigen:
                eigen = self._get_ev(ev)
            if not occ and not vrt and eigen:
                ef = max(eigen)
            if not eigen:
                continue
            for i, e in enumerate(eigen):
                if e:
                    evs[i].append(e)
        for i, energs in enumerate(evs):
            if len(energs) == 0:
                continue
            self.append(np.array(energs), label=label,
                        units='AU', norm=1.0-2.0*i)
        if ef:
            self.fermi_level(ef, units='AU')

    def _get_ev(self, ev, keys=None):
        "Get the correct list of the energies for this eigenvalue"
        res = False
        if keys is None:
            ener = ev.get('e')
            spin = ev.get('s')
            if ener and spin == 1:
                res = [ener]
            elif ener and spin == -1:
                res = [None, ener]
        else:
            for k in keys:
                if k in ev:
                    res = ev[k]
                    if not isinstance(res, list):
                        res = [res]
                    break
        return res

    def append(self, energies, label=None, units='eV', norm=1.0):
        """Include other DoS inside the instance.

        Args:
            energies (list): energies of the superpsition.
            label (str): id of the DoS.
            units (str): units of the energies. Can be AU or eV.
            norm (float, list): weights of the density of states.
        """
        if not isinstance(norm, float) and len(norm) == 0:
            return
        dos = self._conversion_factor(units)*energies
        self.ens.append({'dos': DiracSuperposition(dos, wgts=norm)})
        lbl = label if label is not None else str(len(self.labels)+1)
        self.labels.append(lbl)
        self.set_range()

    def _conversion_factor(self, units):
        if units == 'AU':
            fac = AU_eV
        elif units == 'eV':
            fac = 1.0
        else:
            raise ValueError('Unrecognized units ('+units+')')
        return fac

    def fermi_level(self, fermi_level, units='eV'):
        """ Set the fermi level of the DoS.

        Args:
            fermi_level (float): the value of the chemical potential.
            units (str): the units of this value. can be `AU` or `eV`.
        """
        if fermi_level is not None:
            self.ef = fermi_level*self._conversion_factor(units)

    def set_range(self, npts=None, e_min=None, e_max=None, deltae=None):
        """Adjust the range of the curves to be plotted/represented.

        This function can be called if it is necessary to adjust the range
        of the curves.

        Args:
            npts (int): minimum number of points of the curves.
            deltae (float): minimum resolution of the x range. Always in eV.
            e_min (float): Minimum value of the energy. Always in eV.
            e_max (float): Maximum value of the energy. Always in eV.

        """
        import numpy as np
        npts = npts if npts is not None else getattr(self, 'npts', 2500)
        em = 1.e100
        eM = -1.e100
        demin = 1.e100
        for dos in self.ens:
            mn, mx = dos['dos'].xlim
            em = min(em, mn)
            eM = max(eM, mx)
            demin = min(demin, (eM-em)/npts)

        # reassign windowing
        self.emin = max(
            em, getattr(self, 'emin', -1.e100) if e_min is None else e_min)
        self.emax = min(
            eM, getattr(self, 'emax', 1.e100) if e_max is None else e_max)
        self.deltae = min(demin, getattr(self, 'deltae', 1.e100))
        self.npts = npts
        self.range = np.arange(self.emin, self.emax, self.deltae)

    def get_curves(self, range=None, sigma=None):
        """Get the curves of the DoS instance, ready for plotting.

        This function provides a dictionary of data for each of the curves
        which can be plots by the :py:meth:`plot` method. They can be retrieved
        also to be plotted with other plotting tools.

        Args:
            range (array): the xrange to be employed for the curves.
                If not provided, use the :py:attr:`range` attribute of
                the instance.
            sigma (float): the smearing of the dos. If not provided, use the
                :py:attr:`sigma` attribute of the instance.

        Returns:
            collections.OrderedDict: a dicitonary of a `label: (x,y)` items,
                where x,y are the data to be plotted. (x,y) is a
                `py:class:~collections.namedtuple` class instance,
                as returned from `DiracSuperposition`.
        """
        from collections import OrderedDict
        if range is None:
            xr = self.range
        else:
            xr = range
        sg = sigma if sigma is not None else self.sigma
        return OrderedDict((lb, c['dos'].curve(xr, sg))
                           for lb, c in zip(self.labels, self.ens))

    def shift_curves(self, shift):
        """Shift the curves.

        Apply a constant or a per-curve shift on the data.

        Args:
            shift (float, list, dict): the shift to be implemented.
                If it is a constant, it is performed on all the curves.
                If it is a list, the shif is performed on all the DoS,
                in appending order. Should have the same length of the labels.
                If is is a dict, it assumes the shift per label.
                Only the indicated DoS will be shifted.
                The units of the shift are always in eV.
        """
        if isinstance(shift, dict):
            shiftlist = [shift.get(lb, 0.0) for lb in self.labels]
        elif isinstance(shift, float):
            shiftlist = [shift for lb in self.labels]
        else:
            shiftlist = shift
        for sh, d in zip(shiftlist, self.ens):
            d['dos'].dos += sh

    def dump(self, **kwargs):
        """Represent the dta to be plotted as a table.

        This is a commodity function that can be used to redirect the output
        of the DoS into a file.

        Args:
            **kwargs: the arguments of the :py:meth:`get_curves` method.

        Returns:
            (str): serialized DoS, gnuplot-friendly, can be written in a file.
        """
        curves = self.get_curves(**kwargs)
        allcv = []
        for lb, (x, y) in curves.items():
            xrg = x  # those are all equal
            allcv.append(y)
        buf = ''
        for i, e in enumerate(xrg):
            buf += str(e) + ' ' + ' '.join(map(str, [d[i] for d in allcv]))
            buf += '\n'
        return buf

    def plot(self, sigma=None, ax=None, smearing_slider=False):
        """Commodity Function to plot the DoS.

        Plot the curves which are available.

        Args:
            sigma(float): smearing of the curves
            ax (matplotlib.pyplot.axis): matplotlib axis instance.
                Created if not given.
            smearing_slider (bool): If True, a slider
                is drawn to control the semaring manually.
                Useful when outside the jupyter inline visualization.
        Returns:
            matplotlib.pyplot.axis: The axis instance used for the plot.
                This is also present in the `ax1` attribute of the class.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        if sigma is None:
            sigma = self.sigma
        if ax is None:
            self.fig, self.ax1 = plt.subplots(figsize=(6.4, 4.8))
        else:
            self.fig = ax.get_figure()
            self.ax1 = ax
        curves = self.get_curves(sigma=sigma)
        self.plotl = [self.ax1.plot(x, y, label=lb)
                      for lb, (x, y) in curves.items()]

        self.ax1.set_xlabel('Energy [eV]', fontsize=18)
        self.ax1.set_ylabel('DoS', fontsize=18)
        if self.ef is not None:
            plt.axvline(self.ef, color='k', linestyle='--')
        if smearing_slider:
            axcolor = 'lightgoldenrodyellow'
            try:
                axsigma = plt.axes([0.2, 0.93, 0.65, 0.03], facecolor=axcolor)
            except AttributeError:
                axsigma = plt.axes([0.2, 0.93, 0.65, 0.03], axisbg=axcolor)
            self.ssig = Slider(axsigma, 'Smearing', 0.0, 0.4, valinit=sigma)
            self.ssig.on_changed(self.update)
        if hasattr(self, 'sdos') and self.sdos:
            self._set_sdos_selector()
            self._set_sdos()
        return self.ax1

    def _set_sdos_selector(self):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons
        self.sdos_selector = RadioButtons(
            plt.axes([0.93, 0.05, 0.04, 0.11], axisbg='lightgoldenrodyellow'),
            ('x', 'y', 'z'), active=1)
        self.isdos = 1
        self.sdos_selector.on_clicked(self._update_sdos)

    def _set_sdos(self):
        import numpy
        xs = self.sdos[self.isdos]['coord']
        self._set_sdos_sliders(numpy.min(xs), numpy.max(xs))
        self._update_sdos(0.0)  # fake value as it is unused

    def _sdos_curve(self, sdos, vmin, vmax):
        import numpy
        xs = self.sdos[self.isdos]['coord']
        imin = numpy.argmin(numpy.abs(xs-vmin))
        imax = numpy.argmin(numpy.abs(xs-vmax))
        doslist = sdos[self.isdos]['doslist']
        # norms=self.sdos[self.isdos]['norms'][ispin]
        tocurve = [0.0 for i in doslist[imin]]
        for d in doslist[imin:imax+1]:
            tocurve = [t+dd for t, dd in zip(tocurve, d)]
        # tocurve=numpy.sum([ d[ispin] for d in doslist[imin:imax+1]],axis=0)
        return tocurve
        # float(len(xs))/float(imax+1-imin)*tocurve,norms

    def _update_sdos(self, val):
        isdos = self.isdos
        if val == 'x':
            isdos = 0
        elif val == 'y':
            isdos = 1
        elif val == 'z':
            isdos = 2
        if isdos != self.isdos:
            self.isdos = isdos
            self._set_sdos()

        vmin, vmax = (s.val for s in self.ssdos)
        if vmax < vmin:
            self.ssdos[1].set_val(vmin)
            vmax = vmin
        if vmin > vmax:
            self.ssdos[0].set_val(vmax)
            vmin = vmax
        # now plot the sdos curve associated to the given value
        sig = self.ssig.val
        curves = []
        for dos in self.ens:
            if 'sdos' not in dos:
                continue
            renorms = self._sdos_curve(dos['sdos'], vmin, vmax)
            curve = dos['dos'].curve(self.range, sigma=sig, wgts=renorms)
            curves.append(curve)
        if hasattr(self, '_sdos_plots'):
            for pl, curve in zip(self._sdos_plots, curves):
                pl[0].set_ydata(curve[1])
        else:
            self._sdos_plots = []
            for c in curves:
                self._sdos_plots.append(
                    self.ax1.plot(*c, label='sdos'))
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.fig.canvas.draw_idle()

    def _set_sdos_sliders(self, cmin, cmax):
        import matplotlib.pyplot as plt
        from futile.Figures import VertSlider
        if hasattr(self, 'ssdos'):
            self.ssdos[0].ax.clear()
            self.ssdos[0].__init__(
                self.ssdos[0].ax, 'SDos', cmin, cmax, valinit=cmin)
            self.ssdos[1].ax.clear()
            self.ssdos[1].__init__(self.ssdos[1].ax, '',
                                   cmin, cmax, valinit=cmax)
        else:
            axcolor = 'red'
            axmin = plt.axes([0.93, 0.2, 0.02, 0.65], axisbg=axcolor)
            axmax = plt.axes([0.95, 0.2, 0.02, 0.65], axisbg=axcolor)
            self.ssdos = [
                VertSlider(axmin, 'SDos', cmin, cmax, valinit=cmin),
                VertSlider(axmax, '', cmin, cmax, valinit=cmax)]
        self.ssdos[0].valtext.set_ha('right')
        self.ssdos[1].valtext.set_ha('left')
        self.ssdos[0].on_changed(self._update_sdos)
        self.ssdos[1].on_changed(self._update_sdos)

    def update(self, val):
        sig = self.ssig.val
        curves = self.get_curves(sigma=sig)
        for line, (x, y) in zip(self.plotl, curves.values()):
            line[0].set_ydata(y)
        # for i, dos in enumerate(self.ens):
        #     self.plotl[i][0].set_ydata(
        #         dos['dos'].curve(self.range, sigma=sig)[1])
            # self.plotl[i][0].set_ydata(self.curve(dos,norm=self.norms[i],sigma=sig))
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    import numpy as np
    energies = np.array([-0.815924953235059, -0.803163374736654,
                         -0.780540200987971, -0.7508806541364,
                         -0.723626807289917, -0.714924448617026,
                         -0.710448085701742, -0.68799028016451,
                         -0.67247569974853, -0.659038909236607,
                         -0.625396293324399, -0.608009041659988,
                         -0.565337910777367, -0.561250536074343,
                         -0.551767438323268, -0.541295070404525,
                         -0.532326667587434, -0.515961980147107,
                         -0.474601108285518, -0.473408476151224,
                         -0.46509070541069, -0.445709086452906,
                         -0.433874403837837, -0.416121660651406,
                         -0.407871082254237, -0.406123490618786,
                         -0.403004188319382, -0.38974739285104,
                         -0.380837488456638, -0.375163102271681,
                         -0.375007771592681, -0.367898783582561,
                         -0.367518948507212, -0.359401585874402,
                         -0.358189406008502, -0.354517727598174,
                         -0.334286389724978, -0.332921810616845,
                         -0.315466259109401, -0.308028853904577,
                         -0.29864142362141, -0.294024743731349,
                         -0.292104129933301, -0.285165738729842,
                         -0.28419932605141, -0.267399999874122,
                         -0.259487769142101, -0.239899780812716,
                         -0.224858003804207, -0.20448050758473,
                         -0.164155133452971, -0.117617164459898,
                         -0.0717938081884113, -0.0526986239898579,
                         -0.0346031190163735, -0.0167949342608791,
                         -0.0135168064347152, -0.0102971895842409,
                         0.00759271179427191, 0.00974950976249545,
                         0.010176021051287, 0.0217652761059223,
                         0.0239924727094222, 0.0413057846713024,
                         0.0422334333464529, 0.0459150454793617,
                         0.0517637894860314])
    dos = DoS(energies, fermi_level=-0.1)
    dos.append(0.2+energies)
    dos.dump(sigma=0.01)
    dos.plot()
