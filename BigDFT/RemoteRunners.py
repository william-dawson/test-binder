"""
This module wraps the classes and functions you will need if you want to run
PyBigDFT operations on a remote machine.
"""


from BigDFT.Calculators import Runner
from BigDFT.Datasets import Dataset


VERBOSE = True


def _bind_closure(f, **kwargs):
    """
    Given a function and its named arguments, this routine binds that as
    a lexical context for a closure which is returned.
    """
    return lambda: f(**kwargs)


def _system(cmd):
    from os import system
    if VERBOSE:
        print(cmd)
    system(cmd)


def _ensure_dir(run_dir, ssh=None):
    from os import mkdir
    if ssh is None:
        # Ensure locally
        try:
            mkdir(run_dir)
        except FileExistsError:
            pass
    else:
        # Ensure remotely
        _system(ssh + " mkdir -p " + run_dir)


def _sync_forward(files, rsync, local, url):
    from os.path import join
    for (srcdir, destdir), file_list in files.items():
        allfiles_src = ' '.join([join(srcdir, f) for f in file_list])
        if local:
            _system(rsync + ' ' + allfiles_src + " " + destdir)
        else:
            _system(rsync + ' ' + allfiles_src + " " + url + ":" + destdir)
        #
        # for f in file_list:
        #     src = join(srcdir, f)
        #     dest = join(destdir, f)
        #     if local:
        #         _system(rsync + ' ' + src + " " + dest)
        #     else:
        #         _system(rsync + ' ' + src + " " + url + ":" + dest)


class RemoteFunction():
    """Serialize and execute remotely a python function.

    With this class we serilize and execute a python function
    on a remote filesystem. This filesystem may be a remote computer
    or a different directory of the local machine.

    Such class is useful to be employed in runners which needs to be executed
    outside of the current framework, butfor which the result is needed to
    continue the processing of the data.

    Args:
        name (str): the name of the remote function
        function (func): the python function to be serialized.
            The return value of the function should be an object for which
            a relatively light-weight serialization is possible.
        submitter (str): the interpreter to be invoked. Should be, e.g.
            `python` if the function is a python function, or `bash`.
        **kwargs:  keyword arguments of the function.
            The arguments of the function should be serializable without
            the requirement of too much disk space.
    """
    def __init__(self, submitter, name, **kwargs):
        self.name = name
        self.submitter = submitter
        self.appended_functions = []
        self.arguments = kwargs
        raise NotImplementedError

    def _set_arguments(self):
        pass

    def _serialize(self, run_dir):
        return []

    def _make_run(self, run_dir):
        self.resultfile = None
        self.runfile = None
        raise NotImplementedError

    def _read_results(self, run_dir):
        return None

    def _write_main_runfile(self, pscript, run_dir):
        from os.path import join
        runfile = join(run_dir, self.runfile)
        with open(runfile, "w") as ofile:
            ofile.write(pscript + '\n' + self._append_further_lines())
        return self.runfile

    def _append_further_lines(self):
        pscript = ''
        for func in self.appended_functions:
            pscript += func.call(remotely=False, remote_cwd=True) + '\n'
        return pscript

    # def _sync_forward(self, src, dest):
    #     if self.no_ssh:
    #         _system(self.rsync + ' ' + src + " " + dest)
    #     else:
    #         _system(self.rsync + ' ' + src + " " + self.url + ":" + dest)

    def _append_further_files_to_send(self):
        initials = []
        for func in self.appended_functions:
            # initials += func.prepare_files_to_send(
            #     run_dir=self.local_directory)
            for file_list in func.setup(src=self.local_directory).values():
                initials += file_list
            self.resultfile = func.resultfile  # override resultfile
        return initials

    def files_sent(self, yes):
        """
        Mark the relevant files as sent.

        Args:
            yes (bool): True if the files have been already sent to the
                remote directory
        """
        self.files_have_been_sent = yes

    def prepare_files_to_send(self, run_dir=None):
        """List of files that will have to be sent.

        This function defines the files which requires to be sent over
        in order to execute remotely the python function.
        Also set the main file to be called and the resultfile.

        Args:
            run_dir (str): local directory in which the files will be prepared

        Returns:
            list: files which will be sent
        """
        if run_dir is None:
            run_dir = self.local_directory
        self._set_arguments()
        initials = self._serialize(run_dir)
        pscript = self._make_run(run_dir)
        dependencies = self._append_further_files_to_send()
        self.main_runfile = self._write_main_runfile(pscript, run_dir)
        initials.append(self.main_runfile)
        self.files_sent(False)
        return initials + dependencies

    def setup(self, dest='.', src='/tmp/'):
        """Create and list the files which have to be sent remotely.

        Args:
            dest (str): remote_directory of the destination filesystem.
                The directory should exist and write permission should
                be granted.
            src (str): local directory to prepare the IO files to send.
                The directory should exist and write permission should
                be granted.

        Returns:
            dict: dictionary of {src: dest} files
        """
        self.local_directory = src
        self.remote_directory = dest
        files_to_send = {(self.local_directory, self.remote_directory): [f
                         for f in self.prepare_files_to_send()]}
        self.files_to_send = files_to_send
        return files_to_send

    def send_files(self, files, url='localhost', rsync='rsync -auv'):
        """Send files to the remote filesystem to which the class should run.

        With this function, the relevant serialized files should be sent,
        via the rsync protocol (or equivalent) into the remote directory
        which will then be employed to run the function.

        Args:
            files(dict): source and destination files to be sent,
                organized as items of a dictionary.
            url (str): the remote url to connect to.
            rsync (str): rsync command to synchronize the serialized result.
        """
        self.url = url
        self.no_ssh = url == 'localhost'  # this may need changes for tunnels
        self.rsync = rsync
        if hasattr(self, 'files_have_been_sent') and self.files_have_been_sent:
            pass
        else:
            _sync_forward(files, self.rsync, self.no_ssh, self.url)
        self.files_sent(True)

    def append_function(self, extra_function):
        """Include the call to another function in the main runfile.

        With this method another remote function can be called in the same
        context of the remote call.
        Such remote function will be called from within the same remote
        directory of the overall function.
        The result of the remote function will then be the result provided
        from the last of the appended functions.

        Args:
            extra_function (RemoteFunction): the remote function to append.
        """
        assert extra_function.name != self.name, \
            'Name of the remote function should be different'
        for func in self.appended_functions:
            assert extra_function.name != func.name, \
                'Cannot append functions with the same name'
        self.appended_functions.append(extra_function)

    def send_to(self, dest='.', src='/tmp/', url='localhost',
                rsync='rsync -auv'):
        """Assign the remote filesystem to which the class should run.

        With this function, the relevant serialized files should be sent,
        via the rsync protocol (or equivalent) into the remote directory
        which will then be employed to run the function.

        Args:
            dest (str): remote_directory of the destination filesystem.
                The directory should exist and write permission should
                be granted.
            src (str): local directory to prepare the IO files to send.
                The directory should exist and write permission should
                be granted.
            url (str): the remote url to connect to.
            rsync (str): rsync command to synchronize the serialized result.
        """
        files = self.setup(dest=dest, src=src)
        self.send_files(files, url=url, rsync=rsync)

    def call(self, remotely=True, remote_cwd=False, ssh='ssh'):
        """Provides the command to execute the function.

        Args:
            remotely (bool): invoke the function from local machine.
                employ ssh protocol to perform the call.
            remote_cwd (bool): True if our CWD is already the remote_dir to
                which the function has been sent. False otherwise.
                Always assumed as False if remotely is True.
            ssh (str): the ssh interpreter to be employed.

        Returns:
            str: the string command to be executed.
        """
        command = self.submitter + ' ' + self.main_runfile
        if remotely or not remote_cwd:
            command = 'cd '+self.remote_directory + ' && ' + command
        if remotely and not self.no_ssh:
            self.ssh = ssh
            command = ' '.join([ssh, self.url, '"'+command+'"'])
        return command

    def is_finished(self, remotely=True, ssh='ssh'):
        """
        Check if the function has been executed.
        This is controlled by the presence of the resultfile of the
        last of the functions dependencies.

        Args:
            remotely (bool): control the presence of the result on the remote
                filesystem
            ssh (str): the ssh interpreter to be employed. Use internal value
                if not provided. Only meaningful if remotely is True.
        Return:
            bool: True if ready, False otherwise.
        """
        from subprocess import check_output, CalledProcessError
        from os.path import join

        remotefile = self.resultfile

        ssh = getattr(self, 'ssh', ssh)

        if remotely:
            remotefile = join(self.remote_directory, remotefile)
        else:
            remotefile = join(self.local_directory, remotefile)
        try:
            if not remotely or self.no_ssh:
                check_output(["ls", remotefile])
            else:
                check_output([ssh, self.url, "ls", remotefile])
            return True
        except CalledProcessError:
            return False

    def fetch_result(self, remotely=True):
        """Get the results of a calculation locally.

        Args:
            remotely (bool): control the presence of the result on the remote
                filesystem

        Returns:
            The object returned by the original function
        """
        from os.path import join

        if not self.is_finished(remotely=remotely):
            raise ValueError("Calculation not completed yet")

        # Sync Back
        if self.no_ssh and self.local_directory == self.remote_directory:
            pass
        else:
            _system(self.rsync + ' ' + ('' if self.no_ssh else self.url + ':')
                    + join(self.remote_directory, self.resultfile) +
                    " " + self.local_directory + "/")
        if len(self.appended_functions) > 0:
            func = self.appended_functions[-1]
        else:
            func = self
        # Read results, with dependencies or not
        return func._read_results(self.local_directory)


class RemoteDillFunction(RemoteFunction):
    """Serialize and execute remotely a python function, serialized with dill.

    With this class we serilize and execute a python function
    on a remote filesystem. This filesystem may be a remote computer
    or a different directory of the local machine.

    Such class is useful to be employed in runners which needs to be executed
    outside of the current framework, butfor which the result is needed to
    continue the processing of the data.

    Args:
        name (str): the name of the remote function
        function (func): the python function to be serialized.
            The return value of the function should be an object for which
            a relatively light-weight serialization is possible.
        submitter (str): the interpreter to be invoked. Should be, e.g.
            `python` if the function is a python function, or `bash`.
        required_files (list): list of extra files that may be required for
            the good running of the function.
        **kwargs:  keyword arguments of the function.
            The arguments of the function should be serializable without
            the requirement of too much disk space.
    """
    def __init__(self, submitter, name, function, required_files=[], **kwargs):
        self.name = name
        self.submitter = submitter
        self.appended_functions = []
        self.required_files = []
        self.arguments = kwargs
        self.raw_function = function

    def _set_arguments(self):
        self.function = _bind_closure(self.raw_function, **self.arguments)

    def _serialize(self, run_dir):
        from os.path import join
        from dill import dump
        self.serialized_file = self.name + "-serialize.dill"
        serializedfile = join(run_dir, self.serialized_file)
        with open(serializedfile, "wb") as ofile:
            dump(self.function, ofile, recurse=False)
        return [self.serialized_file] + self.required_files

    def _make_run(self, run_dir):
        self.resultfile = self.name + '-result.dill'
        self.runfile = self.name + "-run.py"
        pscript = "# Run the serialized script\n"
        pscript += "from dill import load, dump\n"
        pscript += "with open(\""+self.serialized_file+"\" , \"rb\")"
        pscript += " as ifile:\n"
        pscript += "    fun = load(ifile)\n"
        pscript += "result = fun()\n"
        pscript += "with open(\""+self.resultfile+"\" , \"wb\") as ofile:\n"
        pscript += "    dump(result, ofile)\n"
        return pscript

    def _read_results(self, run_dir):
        from dill import load
        from os.path import join
        with open(join(run_dir, self.resultfile), "rb") as ifile:
            res = load(ifile)
        return res

    def append_function(self, extra_function):
        raise NotImplementedError('Cannot append a function.' +
                                  ' Use a RemoteScript to concatenate')


class RemoteJSONFunction(RemoteFunction):
    """Serialize and execute remotely a python function, serialized with JSON.

    With this class we serilize and execute a python function
    on a remote filesystem. This filesystem may be a remote computer
    or a different directory of the local machine.

    Such class is useful to be employed in runners which needs to be executed
    outside of the current framework, for which the result is needed locally to
    continue data processing.

    Args:
        name (str): the name of the remote function
        function (func): the python function to be serialized.
            The return value of the function should be an object for which
            a relatively light-weight serialization is possible.
        submitter (str): the interpreter to be invoked. Should be, e.g.
            `python` if the function is a python function, or `bash`.
        extra_encoder_functions (list)): list of dictionaries of the format
            {'cls': Class, 'func': function} which is employed to serialize
            non-instrinsic objects as well as non-numpy objects.
        required_files (list): list of extra files that may be required for
            the good running of the function.
        **kwargs:  keyword arguments of the function.
            The arguments of the function should be serializable without
            the requirement of too much disk space.
    """
    def __init__(self, submitter, name, function, extra_encoder_functions=[],
                 required_files=[], **kwargs):
        from inspect import getsource
        self.name = name
        self.function = getsource(function)
        self.function_name = function.__name__
        self.extra_encoder_functions = extra_encoder_functions
        self.encoder_functions_serialization = {s['cls'].__name__:
                                                {'name': s['func'].__name__,
                                                 'source':
                                                 getsource(s['func'])}
                                                for s in
                                                extra_encoder_functions}
        self.required_files = []
        self.submitter = submitter
        self.appended_functions = []
        self.arguments = kwargs

    def _set_arguments(self):
        from futile.Utils import serialize_objects
        self.arguments_serialization = serialize_objects(
            self.arguments, self.extra_encoder_functions)

    def _serialize(self, run_dir):
        from os.path import join
        from futile.Utils import create_tarball
        if len(self.required_files) + len(self.arguments_serialization) == 0:
            self.serialized_file = ''
            return []
        self.serialized_file = self.name + "-files.tar.gz"
        serializedfile = join(run_dir, self.serialized_file)
        create_tarball(serializedfile, self.required_files,
                       {k+'.json': v
                        for k, v in self.arguments_serialization.items()})
        return [self.serialized_file]

    def _make_run(self, run_dir):
        self.resultfile = self.name + '-result.json'
        self.runfile = self.name + "-run.py"
        pscript = "kwargs = {}\n"
        pscript += "import json\n"
        if self.serialized_file != '':
            pscript += "# Unpack the argument tarfile if present\n"
            pscript += "import tarfile\n"
            pscript += "# extract the archive\n"
            pscript += "arch = tarfile.open('" + self.serialized_file + "')\n"
            pscript += "arch.extractall(path='.')\n"
            pscript += "files = arch.getnames()\n"
            pscript += "arch.close()\n"
            for arg in self.arguments_serialization.keys():
                pscript += "with open('" + arg + ".json', 'r') as f:\n"
                pscript += "    kwargs['" + arg + "'] = json.load(f)\n"
        pscript += "extra_encoder_functions = []\n"
        for name, func in self.encoder_functions_serialization.items():
            pscript += func['source']
            pscript += "extra_encoder_functions.append({'cls'" + name + \
                ", 'func:'" + func['name'] + "})\n"
        pscript += "class CustomEncoder(json.JSONEncoder):\n"
        pscript += "    def default(self, obj):\n"
        pscript += "        try:\n"
        pscript += "            import numpy as np\n"
        pscript += "            nonumpy = False\n"
        pscript += "        except ImportError:\n"
        pscript += "            nonumpy = True\n"
        pscript += "        if not nonumpy:\n"
        pscript += "            if isinstance(obj, (np.int_, np.intc,\n"
        pscript += "                                np.intp, np.int8,\n"
        pscript += "                                np.int16, np.int32,\n"
        pscript += "                                np.int64, np.uint8,\n"
        pscript += "                                np.uint16, np.uint32,\n"
        pscript += "                                np.uint64)):\n"
        pscript += "                return int(obj)\n"
        pscript += "            elif isinstance(obj, (np.float_, np.float16,\n"
        pscript += "                                  np.float32,\n"
        pscript += "                                  np.float64)):\n"
        pscript += "                return float(obj)\n"
        pscript += "            elif isinstance(obj, (np.ndarray,)):\n"
        pscript += "                return obj.tolist()\n"
        pscript += "        if isinstance(obj, (set,)):\n"
        pscript += "            return list(obj)\n"
        pscript += "        else:\n"
        pscript += "            for spec in extra_encoder_functions:\n"
        pscript += "                if isinstance(obj, (spec['cls'],)):\n"
        pscript += "                    return spec['func'](obj)\n"
        pscript += "        return json.JSONEncoder.default(self, obj)\n"
        pscript += self.function
        pscript += "result = "+self.function_name+"(**kwargs)\n"
        pscript += "with open(\""+self.resultfile+"\" , \"w\") as ofile:\n"
        pscript += "    json.dump(result, ofile, cls=CustomEncoder)\n"
        return pscript

    def _read_results(self, run_dir):
        from json import load
        from os.path import join
        with open(join(run_dir, self.resultfile), "r") as ifile:
            res = load(ifile)
        return res

    def append_function(self, extra_function):
        raise NotImplementedError('Cannot append a function.' +
                                  ' Use a RemoteScript to concatenate')


class RemoteScript(RemoteFunction):
    """Triggers the remote execution of a script.

    This class is useful to execute remotely a script and to retrieve.
    The results of such execution. It inherits from the `RemoteFunction`
    base class and extends some of its actions to the concept of the script.

    Args:
        name (str): the name of the remote function
        script (str, func): The script to be executed provided in string form.
            It can also be provided as a function which returns a string.
        result_file(str): the name of the file in which the script
            should redirect.
        submitter (str): the interpreter to be invoked. Should be, e.g.
            `bash` if the script is a shell script, or 'qsub' if this is a
            submission script.
        **kwargs:  keyword arguments of the script-script function,
            which will be substituted in the string representation.
    """
    def __init__(self, submitter, name, script, result_file, **kwargs):
        self.name = name
        self.script = script
        self.submitter = submitter
        self.resultfile = result_file
        self.arguments = kwargs
        self.appended_functions = []

    def _set_arguments(self):
        if isinstance(self.script, str):
            scr = self.script
            for key, value in self.arguments.items():
                scr = scr.replace(key, str(value))
            self.script = scr
        else:
            self.script = self.script(**self.arguments)

    def _read_results(self, run_dir):
        from os.path import join
        with open(join(run_dir, self.resultfile), 'rb') as ifile:
            res = [line.decode('utf-8') for line in ifile.readlines()]
        return res

    def _make_run(self, run_dir):
        self.runfile = self.name + "-run.sh"
        return self.script


class RemoteRunner(Runner, RemoteScript):
    """
    This class can be used to run python functions on a remote machine. This
    class combines the execution of a script with a python remote function.

    Args:

        function (func): the python function to be serialized.
            The return value of the function should be an object for which
            a relatively light-weight serialization is possible.
        name (str): the name of the remote function
        script (str): The script to be executed provided in string form.
            The result file of the script is assumed to be named
            `<name>-script-result`.
        submitter (str): the interpreter to be invoked. Should be, e.g.
            `bash` if the script is a shell script, or `qsub` if this is a
            submission script.
        url (str): the url you would use to ssh into that remote machine.
        remote_dir (str): the path to the work directory on the remote machine.
           Should be associated to a writable directory.
        skip (bool): if true, we perform a lazy calculation.
        asynchronous (bool): If True, submit the calculation without waiting
           for the results.
        local_dir (str): local directory to prepare the IO files to send.
            The directory should exist and write permission should
            be granted.
        python (str): python interpreter to be invoked in the script.
        protocol (str): serialization method to be invoked for the function.
            can be 'JSON' of 'Dill', depending of the desired version.
        extra_encoder_functions (list)): list of dictionaries of the format
            {'cls': Class, 'func': function} which is employed to serialize
            non-instrinsic objects as well as non-numpy objects. Useful for
            the 'JSON' protocol.
        required_files (list): list of extra files that may be required for
            the good running of the function.
        arguments (dict):  keyword arguments of the function.
            The arguments of the function should be serializable without
            the requirement of too much disk space.
        **kwargs (dict):  Further keyword arguments of the script,
            which will be substituted in the string representation.
    """

    def __init__(self, function, submitter='bash', name='remotefunction',
                 url='localhost', skip=True, asynchronous=True, remote_dir='.',
                 ssh='ssh', rsync='rsync -auv', local_dir='/tmp',
                 script="#!/bin/bash\n", python='python', arguments={},
                 protocol='JSON', extra_encoder_functions=[],
                 required_files=[], **kwargs):
        super().__init__(submitter=submitter, name=name, script=script,
                         result_file=name + '-script-result',
                         url=url, skip=skip, asynchronous=asynchronous,
                         remote_dir=remote_dir,
                         ssh=ssh, rsync=rsync, local_dir=local_dir,
                         protocol=protocol, python=python,
                         extra_encoder_functions=extra_encoder_functions,
                         required_files=required_files, function=function,
                         arguments=arguments, **kwargs)
        self.remote_function = self._create_remote_function(
            name, self._global_options)
        self.append_function(self.remote_function)

    def _create_remote_function(self, name, options):
        rfargs = {'submitter': options['python'],
                  'name': name + '-function',
                  'function': options['function'],
                  'required_files': options['required_files']}
        protocol = options['protocol']
        if protocol == 'JSON':
            rfargs.update(options['extra_encoder_functions'])
            cls = RemoteJSONFunction
            pass
        elif protocol == 'Dill':
            cls = RemoteDillFunction
            pass
        rfargs.update(options['arguments'])
        return cls(**rfargs)

    def pre_processing(self):
        self.remote_function.arguments = self.run_options['arguments']
        if hasattr(self, 'files_to_send'):
            files_to_send = self.files_to_send
        else:
            files_to_send = self.setup(dest=self.run_options['remote_dir'],
                                       src=self.run_options['local_dir'])
        return {'files': files_to_send}

    def _get_opts(self, opts):
        return {key: self.run_options[key] for key in opts}

    def process_run(self, files):
        if self.run_options['skip']:
            if self.is_finished(remotely=False):
                return {'status': 'finished_locally'}
        self.send_files(files, **self._get_opts(['url', 'rsync']))
        ssh = self.run_options['ssh']
        if self.run_options['skip']:
            if self.is_finished(ssh=ssh):
                return {'status': 'finished_remotely'}
        command = self.call(ssh=ssh)
        if self.run_options['asynchronous'] and self.no_ssh:
            command += ' &'
        _system(command)
        return {'status': 'submitted'}

    def post_processing(self, files, status):
        if self.run_options['asynchronous']:
            if status == 'finished_locally':
                return self.fetch_result(remotely=False)
        else:
            return self.fetch_result()


class RemoteDataset(Dataset):
    """
    Defines a set of remote runs, to be executed from a base script and to
    a provided url. This class is associated to a set of remote submissions,
    which may contain multiple calculations. All those calculations are
    expressed to a single url, with a single base script, but with a collection
    of multiple remote runners that may provide different arguments.

    Args:
        label (str): man label of the dataset.
        run_dir (str): local directory of preparation of the data.
        database_file (str): name of the database file to keep track of the
            submitted runs.
        force (str): force the execution of the dataset regardless of the
            database status.
        **kwargs: global arguments of the appended remote runners.
    """
    def __init__(self, label='RemoteDataset', run_dir='/tmp',
                 database_file='database.yaml', force=False,
                 **kwargs):
        Dataset.__init__(self, label=label, run_dir=run_dir, force=force,
                         **kwargs)
        self.database_file = database_file
        self.database = self._construct_database(self.database_file)

    def append_run(self, id, remote_runner=None, **kwargs):
        """Add a remote run into the dataset.

        Append to the list of runs to be performed the corresponding runner and
           the arguments which are associated to it.

        Args:
          id (dict): the id of the run, useful to identify the run in the
             dataset. It has to be a dictionary as it may contain
             different keyword. For example a run might be classified as
             ``id = {'hgrid':0.35, 'crmult': 5}``.
          remote_runner (RemoteRunner): a instance of a remote runner that will
              be employed.
          **kwargs: arguments required for the creation of the corresponding
              remote runner. If remote_runner is provided, these arguments will
              be They will be combined with the global arguments.

        Raises:
          ValueError: if the provided id is identical to another previously
             appended run.
        """
        from os.path import join
        # first fill the internal database
        Dataset.append_run(self, id, Runner(), **kwargs)
        # then create the actual remote runner to be included
        inp = self.runs[-1]
        local_dir = inp.get('local_dir')
        if local_dir is not None:
            basedir = join(self.get_global_option('run_dir'), local_dir)
        else:
            basedir = self.get_global_option('run_dir')
        inp['local_dir'] = basedir
        name = self.names[-1]
        # the arguments have to be substituted before the run call
        remote_script = RemoteRunner(name=name, **inp)
        self.calculators[-1]['calc'] = remote_script

    def pre_processing(self):
        from warnings import warn

        def get_info_from_runner(irun, info):
            run_inp = self.runs[irun]
            val = run_inp.get(info, remote_runner._global_options.get(info))
            return val

        def get_local_run_info(irun, info):
            return self.run_options.get(info, get_info_from_runner(irun, info))

        # gather all the data to be sent
        files_to_send = {}
        selection = []
        force = self.run_options['force']
        for irun, (name, calc) in enumerate(zip(self.names, self.calculators)):
            # Check the database.
            do_it = True
            if not force:
                if self._check_database(name):
                    run_dir = self.get_global_option('run_dir')
                    warn(str((name, run_dir)) + " already submitted",
                         UserWarning)
                    do_it = False
            if do_it:
                selection.append(irun)
            else:
                continue
            remote_runner = calc['calc']
            local_dir = remote_runner.get_global_option('local_dir')
            _ensure_dir(local_dir)
            remote_dir = get_info_from_runner(irun, 'remote_dir')
            fs = remote_runner.setup(dest=remote_dir, src=local_dir)
            for srcdest, files in fs.items():
                for f in files:
                    files_to_send.setdefault(srcdest, []).append(f)
        # then send the data, by only employing the first runner
        # and mark them as sent.
        for irun, calc in enumerate(self.calculators):
            remote_runner = self.calculators[irun]['calc']
            if irun > 0:
                remote_runner.files_sent(True)
            remote_runner.send_files(files_to_send,
                                     url=get_local_run_info(irun, 'url'),
                                     rsync=get_local_run_info(irun, 'rsync'))
        self.selection = selection
        return {}

    def process_run(self):
        """
        Run the dataset, by performing explicit run of each of the item of the
           runs_list.
        """
        self._run_the_calculations(selection=self.selection)
        # Also, update the database informing that
        # the data should run by now.
        for irun, name in enumerate(self.names):
            if irun not in self.selection:
                continue
            self._register_database(name)
        return {}

    def _check_database(self, name):
        return name in self.database

    def _register_database(self, name):
        if name not in self.database:
            self.database.append(name)
            self._write_database()

    def _construct_database(self, name):
        from yaml import load, Loader
        try:
            with open(name, "r") as ifile:
                return load(ifile, Loader=Loader)
        except FileNotFoundError:
            return []

    def clean_database(self):
        """Remove the database information."""
        for name in list(self.database):
            self._remove_database_entry(name)

    def _remove_database_entry(self, name):
        if name in self.database:
            self.database.remove(name)
        self._write_database()

    def _write_database(self):
        from yaml import dump
        # Update the file
        with open(self.database_file, "w") as ofile:
            dump(self.database, ofile)


# class RemoteRunner():
#     """
#     This class can be used to run python functions on a remote machine. This
#     class should not be used as is. Instead, you need to inherit from it, and
#     define the following function: `make_script`.
#
#     A database file is stored locally which keeps tracks of jobs you have
#     submitted. In this way, if you close your notebook/script and re-run it,
#     jobs won't be resubmitted.
#
#     url (str): the url you would use to ssh into that remote machine.
#     remote_dir (str): the full, absolute path to the work directory on
#       the remote machine.
#     database (str): the name of the database file to store locally.
#     skip (bool): if true, we perform a lazy calculation.
#     submitter (str): the command used for running the calculation (bash, qsub,
#       etc).
#     """
#     ssh = 'ssh '
#     rsync = 'rsync -auv '
#
#     def __init__(self, url, remote_dir, database, skip=False,
#                  submitter="bash"):
#         self.url = url
#         self.remote_dir = remote_dir
#         self.submitter = submitter
#         self.skip = skip
#         self.database_file = database
#         self.database = self._construct_database(database)
#         # Because we will get None if the set is empty.
#         if self.database is None:
#             self.database = []
#         if self.url not in ['', 'localhost']:
#             self.connection_command = self.ssh + self.url
#             self.rsync_command = self.rsync + self.url + ":"
#         else:
#             self.connection_command = ''
#             self.rsync_command = self.rsync
#
#     def run(self, function, kwargs, name, force=False, run_dir=".",
#             make_script_args={}, submitter_args=""):
#         """
#         Run a python function on a remote machine. This will launch
#         the job asynchronously.
#
#         function (function): the function to run.
#         kwargs (dict): keyword arguments to be passed to the function
#         name (str): a unique name for this job.
#         run_dir (str): the name of a scratch directory to work in.
#         make_script_args (dict): any extra args that should be passed to the
#           `make_script` command.
#         submitter_args (str): any extra arguments you want to pass to the
#           submission command.
#         """
#         from os.path import join
#         from warnings import warn
#
#         # Check lazy calculation
#         if self.skip:
#             if self.is_finished(name, run_dir):
#                 return
#
#         # Check the database.
#         if not force:
#             if self._check_database(name):
#                 warn(str((name, run_dir)) + " already submitted", UserWarning)
#                 return
#
#         # create the function to bind
#         bind_function = _bind_closure(f=function, **kwargs)
#
#         # Write the files
#         self._ensure_dir(run_dir)
#         self._serialize(bind_function, name, run_dir)
#         self._make_run(name, run_dir)
#         self._make_steps(name, run_dir, submitter_args)
#
#         scr = self.make_script(name, **make_script_args)
#         with open(join(run_dir, name + "-run.sh"), "w") as ofile:
#             ofile.write(scr)
#
#         # Sync Forward
#         self._sync_forward(name + "-serialize.dill", run_dir)
#         self._sync_forward(name + "-run.py", run_dir)
#         self._sync_forward(name + "-run.sh", run_dir)
#
#         # Remote Run
#         if self.connection_command != '':
#             self._system("cat " + join(run_dir, name+"-steps.sh") +
#                          " | " + self.connection_command)
#         else:
#             self._system("sh " + join(run_dir, name+"-steps.sh") + ' & ')
#
#         # then register database if everything OK
#         self._register_database(name)
#
#     def is_finished(self, name, run_dir="."):
#         """
#         Check if the remote calculation has finished.
#
#         Args:
#             name (str): the name of the calculation that was run.
#         """
#         from subprocess import check_output, CalledProcessError
#         from os.path import join
#
#         try:
#             if self.connection_command != '':
#                 check_output([self.ssh, self.url, "ls",
#                               join(self.remote_dir, run_dir,
#                               name + "-result.dill")])
#             else:
#                 check_output(["ls", join(self.remote_dir, run_dir,
#                               name + "-result.dill")])
#             return True
#         except CalledProcessError:
#             return False
#
#     def block_all(self, names, run_dir="."):
#         """
#         Block until all of the calculations in a list are completed.
#
#         Args:
#             names (list): a list of calculation names.
#             run_dir (str): the directory the calculations were run in.
#
#         Returns:
#             A dictionary mapping names to results.
#         """
#         from time import sleep
#         results = {}
#         while any([x not in results for x in names]):
#             for n in names:
#                 if n in results:
#                     continue
#                 if self.is_finished(n, run_dir):
#                     results[n] = self.get_results(n, run_dir)
#             sleep(5.0)
#
#         return results
#
#     def get_results(self, name, run_dir="."):
#         """
#         Get the results of a calculation locally.
#
#         name (str): the name of the calculation that was run.
#         run_dir (str): the work directory.
#
#         Returns:
#           Whatever object that was created remotely.
#         """
#         from os.path import join
#
#         if not self.is_finished(name, run_dir):
#             raise ValueError("Calculation not completed yet")
#
#         # Sync Back
#         self._system(self.rsync_command +
#                      join(self.remote_dir, run_dir, name + "-result.dill") + " " +
#                      run_dir + "/")
#
#         # Clear the database
#         # self._remove_database_entry(name)
#
#         # Read results
#         return self._read_results(name, run_dir)
#
#     def _check_database(self, name):
#         return name in self.database
#
#     def _register_database(self, name):
#         if name not in self.database:
#             self.database.append(name)
#             self._write_database()
#
#     def _construct_database(self, name):
#         from yaml import load, Loader
#         try:
#             with open(name, "r") as ifile:
#                 return load(ifile, Loader=Loader)
#         except FileNotFoundError:
#             return []
#
#     def clean_database(self):
#         """Remove the database information."""
#         for name in list(self.database):
#             self._remove_database_entry(name)
#
#     def _remove_database_entry(self, name):
#         if name in self.database:
#             self.database.remove(name)
#         self._write_database()
#
#     def _write_database(self):
#         from yaml import dump
#         # Update the file
#         with open(self.database_file, "w") as ofile:
#             dump(self.database, ofile)
#
#     def _system(self, cmd):
#         from os import system
#         print (cmd)
#         system(cmd)
#
#     def _ensure_dir(self, run_dir):
#         from os import mkdir
#         from os.path import join
#
#         # Ensure locally
#         try:
#             mkdir(run_dir)
#         except FileExistsError:
#             pass
#
#         # Ensure remotely
#         self._system(self.connection_command + " mkdir -p " +
#                      join(self.remote_dir, run_dir))
#
#     def _sync_forward(self, name, run_dir):
#         from os.path import join
#         if self.connection_command != '':
#             self._system(self.rsync + join(run_dir, name) + " " +
#                          self.url + ":" + join(self.remote_dir, run_dir, name))
#         else:
#             self._system(self.rsync + join(run_dir, name) + " " +
#                          join(self.remote_dir, run_dir, name))
#
#     def _serialize(self, function, name, run_dir):
#         from os.path import join
#         from dill import dump
#         with open(join(run_dir, name + "-serialize.dill"), "wb") as ofile:
#             dump(function, ofile, recurse=False)
#
#     def _make_run(self, name, run_dir):
#         from os.path import join
#         pscript = "# Run the serialized script\n"
#         pscript += "from dill import load, dump\n"
#         pscript += "with open('" + name + "-serialize.dill', \"rb\")"
#         pscript += " as ifile:\n"
#         pscript += "    fun = load(ifile)\n"
#         pscript += "result = fun()\n"
#         pscript += "with open('" + name + "-result.dill', \"wb\") as ofile:\n"
#         pscript += "    dump(result, ofile)\n"
#
#         with open(join(run_dir, name+"-run.py"), "w") as ofile:
#             ofile.write(pscript)
#
#     def make_script(self, name, **kwargs):
#         """
#         This function should be overloaded. In here you need to create a string
#         which is the set of commands to be run. It should end with a call
#         to python name-run.py, where name is the name argument above. You
#         might need to change the name of the python executable, run various
#
#         Example:
#             scr += 'eval "$(conda shell.bash hook)"\n'
#             scr += "conda activate bigdft\n"
#             scr += "source ~/binaries/base/install/bin/bigdftvars.sh\n"
#             scr += "python " + name + "-run.py" + "\n"
#             return scr
#
#         Args:
#             name (str):
#         Returns:
#             (str): a string which is the text of the jobscript to submit.
#         """
#         raise NotImplementedError()
#
#     def _make_steps(self, name, run_dir, submitter_args):
#         from os.path import join
#         steps = "cd " + self.remote_dir + "\n"
#         steps += "cd " + run_dir + "\n"
#         steps += self.submitter + " " + submitter_args + " " + \
#             name + "-run.sh\n"
#         with open(join(run_dir, name + "-steps.sh"), "w") as ofile:
#             ofile.write(steps)
#
#     def _read_results(self, name, run_dir):
#         from dill import load
#         from os.path import join
#         with open(join(run_dir, name + "-result.dill"), "rb") as ifile:
#             res = load(ifile)
#         return res
