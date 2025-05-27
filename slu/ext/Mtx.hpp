#pragma once
#ifndef _HGUARD__A_CPU_A__Mtx_hpp__LIB
#define _HGUARD__A_CPU_A__Mtx_hpp__LIB
/*
MIT License

Copyright (c) 2025 a-cpu-a

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <mutex>
#include <shared_mutex>

template<class T>
struct Mutex
{
	std::mutex lock;
	T v;

	Mutex(const T& v) :v(v) {}
	Mutex(T&& v) :v(std::move(v)) {}
	Mutex() =default;
};

template<class T>
struct RwLock
{
	std::shared_mutex lock;
	T v;

	RwLock(const T& v) :v(v) {}
	RwLock(T&& v) :v(std::move(v)) {}
	RwLock() =default;
};
#endif
