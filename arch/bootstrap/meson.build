if not get_option('no-binary')
    py.extension_module(
        '_samplers',
        '_samplers.pyx',
        install: true,
        include_directories: [inc_np],
        subdir: 'arch/bootstrap',
        c_args: cython_c_args,
        cython_args: cython_args,
    )
endif
