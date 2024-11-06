/*
** See Copyright Notice at the end of this file
*/
#pragma once

#include <iostream>

#ifndef LUA_INCL_IMPLEMENTATION
#define LUA_INCL_IMPLEMENTATION
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 0
#endif

#define block _block_from_lua //avoid name conflicts
#ifndef NDEBUG
#define LUAI_ASSERT
#endif // !NDEBUG


extern "C"
{

#include <luaconf.h>

#include <ldebug.h>
#include <ltable.h>
#include <lopcodes.h>
#include <lobject.h>
#include <lstring.h>
#include <llimits.h>
#include <lctype.h>
#include <llex.h>
#include <lmem.h>
#include <lstate.h>
#include <lzio.h>
#include <lparser.h>
#include <lfunc.h>
#include <lundump.h>
#include <lcode.h>
#include <ldo.h>
#include <ltm.h>
#include <lgc.h>
#include <lvm.h>
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
#include <lapi.h>

#ifdef INCLUDE_LUA_C_FILES
#include <lstrlib.c>
#include <lmathlib.c>
#include <lcorolib.c>
#include <lbaselib.c>
#include <ldblib.c>//debug
#include <ltablib.c>
#include <lutf8lib.c>
#include <loslib.c>

#include <ldebug.c>
#include <ltable.c>
#include <lopcodes.c>
#include <lobject.c>
#include <lstring.c>
#include <lctype.c>
#include <llex.c>
#include <lmem.c>
#include <lstate.c>
#include <lzio.c>
#include <lparser.c>
#include <lfunc.c>
#include <lundump.c>
#include <ldump.c>
#include <lcode.c>
#include <linit.c>
#include <ldo.c>
#include <ltm.c>
#include <lgc.c>
#include <lvm.c>
#include <lauxlib.c>
#include <lapi.c>
#endif

}

#undef block
#undef cast
#undef zgetc
#undef next
#if _XOPEN_SOURCE==0
#undef _XOPEN_SOURCE
#endif
//make sure its not defined ^^^ (it might break stuff)

#endif



/******************************************************************************
* Copyright (C) 2024 a-cpu-a
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/