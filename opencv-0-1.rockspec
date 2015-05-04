package = "opencv"
version = "0-1"

source = {
   url = "https://github.com/Saulzar/lua---opencv.git",
}

description = {
   summary = "OpenCV bindings for Torch",
   detailed = [[
   ]],
   homepage = "https://github.com/Saulzar/lua---opencv",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}