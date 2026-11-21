#include "maix_basic.hpp"

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    maix::sys::register_default_signal_handle();
    return 0;
}
