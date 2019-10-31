function [sequences, experiments] = workspace_load(pwd)
% workspace_load Initializes the current workspace
%
% This function initializes the current workspace by reading sequences and
% experiments as well as initializing global variables. It also checks if native
% resources have to be downloaded or compiled.
%
% To make loading faster when running the script multiple times, it checks if
% sequences and experiments variables exist in the workspace and if they are
% cell arrays and just reuses them. No further check is performed so this may
% lead to problems when switching workspaces. Clear the workspace or use 'Force'
% argument to avoid issues with cached data.
%
% Input:
% - varargin[Force] (boolean): Force reloading the sequences and experiments.
% - varargin[Directory] (string): Set the directory of the workspace (otherwise
%   current directory is used.
% - varargin[OnlyDefaults] (boolean): Only set default global variables and skip
%   workspace initialization.
%
% Output:
% - sequences (cell): Array of sequence structures.
% - experiments (cell): Array of experiment structures.
%

force = false;
directory = pwd;
only_defaults = false;

args = varargin;
for j=1:2:length(args)
    switch lower(varargin{j})
        case 'directory', directory = args{j+1};
        case 'force', force = args{j+1};
        case 'onlydefaults', only_defaults = args{j+1};
        otherwise, error(['unrecognized argument ' args{j}]);
    end
end

if ~force

    try

        % Attempts to load variables from the workspace namespace to save
        % some time
        sequences = evalin('base', 'sequences');

        % Are variables correts at a glance ...
        cached = iscell(sequences);

    catch

        cached = false;

    end

else

    cached = false;

end;

% Some defaults
set_global_variable('toolkit_path', fileparts(fileparts(mfilename('fullpath'))));
set_global_variable('indent', 0);
set_global_variable('directory', directory);
set_global_variable('debug', 0);
set_global_variable('cache', 1);
set_global_variable('bundle', []);
set_global_variable('cleanup', 1);
set_global_variable('trax_mex', []);
set_global_variable('trax_client', []);
set_global_variable('trax_timeout', 30);
set_global_variable('matlab_startup_model', [923.5042, -4.2525]);
set_global_variable('legacy_rasterization', false);
set_global_variable('native_path', fullfile(get_global_variable('toolkit_path'), 'native'));

if only_defaults
	sequences = {};
	experiments = {};
	return;
end;

configuration_file = fullfile(directory, 'configuration.m');

if ~isascii(get_global_variable('toolkit_path'))
    warning('Toolkit path contains non-ASCII characters. This may cause problems.')
end;

if ~isascii(directory)
    warning('Workspace path contains non-ASCII characters. This may cause problems.')
end;

if ~exist(configuration_file, 'file')
    error('Directory is probably not a VOT workspace. Please run workspace_create first.');
end;

print_text('Initializing workspace ...');

configuration_script = get_global_variable('select_configuration', 'configuration');

if ~strcmp(directory, pwd())
    addpath(directory);
end;

try
	environment_configuration = str2func(configuration_script);
    environment_configuration();
catch e
    if exist(configuration_script) ~= 2 %#ok<EXIST>
        print_debug('Global configuration file does not exist. Using defaults.', ...
            configuration_script);
    else
        error(e);
    end;
end;

% Check for potential updates
if get_global_variable('check_updates', true)
    check_updates();
end;

mkpath(get_global_variable('native_path'));
rmpath(get_global_variable('native_path')); rehash; % Try to avoid locked files on Windows
initialize_native();
addpath(get_global_variable('native_path'));

experiment_stack = get_global_variable('stack', 'vot2013');

if exist(['stack_', experiment_stack]) ~= 2 %#ok<EXIST>
    error('Experiment stack %s not available.', experiment_stack);
end;

stack_configuration = str2func(['stack_', experiment_stack]);

experiments = stack_configuration();

if cached
    print_debug('Skipping loading sequence data (using cached structures)');
else

    sequences_directory = get_global_variable('sequences_path', fullfile(get_global_variable('workspace_path'), 'sequences'));

    print_text('Loading sequences ...');

    sequences = sequence_load(sequences_directory, ...
        'list', get_global_variable('sequences', 'list.txt'));

    if isempty(sequences)
        error('No sequences available. Stopping.');
    end;

end;

end

function [updated, latest] = check_updates()
% Check for toolkit updates online.

updated = false;
latest = [];

toolkit_path = get_global_variable('toolkit_path');
workspace_path = get_global_variable('workspace_path');

mkpath(fullfile(workspace_path, 'cache'));
timestamp_file = fullfile(workspace_path, 'cache', '.update_check');

check_interval = get_global_variable('update_check_interval', 0.1);

current_timestamp = datenum(clock());
previous_timestamp = 0;

fd = fopen(timestamp_file, 'r');
if fd > 0
    previous_timestamp = fscanf(fd, '%f');
    fclose(fd);
end

if current_timestamp > previous_timestamp + check_interval

    if exist(fullfile(toolkit_path, 'PACKAGE'), 'file')
        package = fileread(fullfile(toolkit_path, 'PACKAGE'));
        print_text('Checking for toolkit updates on BinTray.')
        latest = get_bintray_version(package);
    else
        print_text('Checking for toolkit updates on GitHub.')
        latest = get_github_version();
    end;

    if isempty(latest)
        updated = false;
        return;
    end;

    updated = toolkit_version(latest) > 0;

    fd = fopen(timestamp_file, 'w'); fprintf(fd, '%f', current_timestamp); fclose(fd);

    if updated
        version = toolkit_version();
        version = sprintf('%d.%d.%d', version.major, version.minor, version.patch);
        print_text('');
        print_text('***************************************************************************');
        print_text('');
        print_text('                  *** Toolkit update available ***');
        print_text('');
        print_text('The VOT toolkit has been updated, a new version is available online. Please');
        print_text('consider updating your local copy or consult the release log for more');
        print_text('information.');
        print_text('');
        print_text('Local version %s, remote version %s', version, latest);
        print_text('');
        print_text('***************************************************************************');
        print_text('');

    end

end;

end

function version = get_bintray_version(package)

    api_server = 'https://api.bintray.com/';

    try
        meta = json_decode(urlread(sprintf('%s/packages/votchallenge/toolkit/%s', api_server, package)));
		version = meta.latest_version;
    catch
        version = [];
        return;
    end

end

function version = get_github_version()

    try
        version = urlread('https://raw.githubusercontent.com/votchallenge/vot-toolkit/master/VERSION');
    catch
        version = [];
        return;
    end

end

