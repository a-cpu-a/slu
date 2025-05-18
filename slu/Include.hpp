/*
** See Copyright Notice at the end of this file
*/
#pragma once

//hg
#ifndef _SLU__INCLUDE_HPP
#define _SLU__INCLUDE_HPP


#include <iostream>
#include <exception>

#include <slu/ErrorType.hpp>

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

#include <slu/ext/lua/luaconf.h>

#include <slu/ext/lua/ldebug.h>
#include <slu/ext/lua/ltable.h>
#include <slu/ext/lua/lopcodes.h>
#include <slu/ext/lua/lobject.h>
#include <slu/ext/lua/lstring.h>
#include <slu/ext/lua/llimits.h>
#include <slu/ext/lua/lctype.h>
#include <slu/ext/lua/llex.h>
#include <slu/ext/lua/lmem.h>
#include <slu/ext/lua/lstate.h>
#include <slu/ext/lua/lzio.h>
#include <slu/ext/lua/lparser.h>
#include <slu/ext/lua/lfunc.h>
#include <slu/ext/lua/lundump.h>
#include <slu/ext/lua/lcode.h>
#include <slu/ext/lua/ldo.h>
#include <slu/ext/lua/ltm.h>
#include <slu/ext/lua/lgc.h>
#include <slu/ext/lua/lvm.h>
#include <slu/ext/lua/lua.h>
#include <slu/ext/lua/lualib.h>
#include <slu/ext/lua/lauxlib.h>
#include <slu/ext/lua/lapi.h>

#ifdef INCLUDE_LUA_C_FILES
#include <slu/ext/lua/lstrlib.c>
#include <slu/ext/lua/lmathlib.c>
#include <slu/ext/lua/lcorolib.c>
#include <slu/ext/lua/lbaselib.c>
#include <slu/ext/lua/ldblib.c>//debug
#include <slu/ext/lua/ltablib.c>
#include <slu/ext/lua/lutf8lib.c>
#include <slu/ext/lua/loslib.c>

#include <slu/ext/lua/ldebug.c>
#include <slu/ext/lua/ltable.c>
#include <slu/ext/lua/lopcodes.c>
#include <slu/ext/lua/lobject.c>
#include <slu/ext/lua/lstring.c>
#include <slu/ext/lua/lctype.c>
#include <slu/ext/lua/llex.c>
#include <slu/ext/lua/lmem.c>
#include <slu/ext/lua/lstate.c>
#include <slu/ext/lua/lzio.c>
#include <slu/ext/lua/lparser.c>
#include <slu/ext/lua/lfunc.c>
#include <slu/ext/lua/lundump.c>
#include <slu/ext/lua/ldump.c>
#include <slu/ext/lua/lcode.c>
#include <slu/ext/lua/linit.c>
#include <slu/ext/lua/ldo.c>
#include <slu/ext/lua/ltm.c>
#include <slu/ext/lua/lgc.c>
#include <slu/ext/lua/lvm.c>
#include <slu/ext/lua/lauxlib.c>
#include <slu/ext/lua/lapi.c>
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

#endif // _SLU__INCLUDE_HPP