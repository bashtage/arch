if not get_option('no-binary')
    py.extension_module(
        'recursions',
        'recursions.pyx',
        install: true,
        include_directories: [inc_np],
        subdir: 'arch/univariate',
        c_args: cython_c_args,
    )
endif
