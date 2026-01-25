/*
    A program file.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This file is part of Slu-c.

    Slu-c is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Slu-c is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with Slu-c.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <print>
#include <span>
#include <unordered_set>
#include <vector>

#include <slu/Ansi.hpp>
export module zzz_internal_slu.test;

import slu.settings;
import slu.ast.mp_data;
import slu.ast.state;
import slu.gen.gen;
import slu.gen.output;
import slu.paint.basic_col_conv;
import slu.paint.paint;
import slu.paint.sem_output;
import slu.paint.to_html;
import slu.parse.error;
import slu.parse.parse;
import slu.parse.vec_input;

using namespace std::string_view_literals;

template<class... Args>
inline void log(const std::format_string<Args...> fmt, Args&&... fmtArgs)
{
	std::cout << LUACC_DEFAULT
	          << std::vformat(fmt.get(), std::make_format_args(fmtArgs...))
	          << "\n";
}
template<class... Args>
inline void logErr(const std::format_string<Args...> fmt, Args&&... fmtArgs)
{
	std::cerr << LUACC_INVALID
	          << std::vformat(fmt.get(), std::make_format_args(fmtArgs...))
	          << "\n";
}

inline std::vector<uint8_t> getBin(const std::string& path)
{
	std::ifstream in(path, std::ios::binary);
	if (!in)
	{
		logErr("Failed to open file for reading: {}", path);
		return {};
	}
	std::vector<uint8_t> data;
	while (!in.eof())
	{
		const int dat = in.get();
		if (!in.eof())
			data.push_back((uint8_t)dat);
	}
	return data;
}

inline void saveBin(const std::string& path, const std::vector<uint8_t>& data)
{
	std::filesystem::create_directories(
	    std::filesystem::path(path).parent_path());
	std::ofstream out(path, std::ios::binary);
	if (!out)
	{
		logErr("Failed to open file for writing: {}", path);
		return;
	}
	out.write(reinterpret_cast<const char*>(data.data()), data.size());
}

inline uint8_t testSluOnFile(
    const std::filesystem::path& path, const bool invert)
{
	const bool wrapItInFn = path.stem().string().ends_with(".$block");
	//fn _(){...}
	if (wrapItInFn)
		return 2; //TODO: implement it
	std::string pathStr = path.string();
	std::vector<uint8_t> srcCode = getBin(pathStr);

	//handle the lua bash thing
	if (srcCode[0] == '#')
	{
		size_t i = 0;
		while (true)
		{
			if (srcCode[i++] == '\n')
				break;
		}
		srcCode.erase(srcCode.begin(), srcCode.begin() + i - 1);
	}

	using Settings = decltype(slu::parse::sluCommon);
	using ColConv = slu::paint::BasicColorConverter<false>;


	slu::parse::BasicMpDbData dbData;
	slu::parse::VecInput<Settings> in;
	in.fName = pathStr;
	in.text = srcCode;
	in.genData.mpDb.data = &dbData;
	in.genData.totalMp = {"testmp"};

	try
	{
		//Slu_assert(!path.filename().string().ends_with("concatenated.slu"));

		slu::parse::ParsedFile f = slu::parse::parseFile(in);


		if (invert)
		{
			throw std::runtime_error("Test failed");
		} else
			log("Test success : {}", pathStr);

		slu::paint::SemOutput<decltype(in), ColConv> se = {in};
		in.restart();
		slu::paint::paintFile(se, f);
		in.restart();
		const std::string outHtml = slu::paint::toHtml(se, true);

		slu::parse::Output<Settings> out;
		out.db = std::move(in.genData.mpDb);
		slu::parse::genFile(out, f);


		saveBin((path.parent_path() / "_TEST_OUT"
		            / (path.filename().string() + ".d"))
		            .string(),
		    out.text);

		slu::parse::VecInput<Settings> in2;
		in2.genData.mpDb.data = &dbData;
		in2.genData.totalMp = {"testmp"};
		in2.fName = pathStr;
		in2.text = out.text;
		in2.restart();

		try
		{

			slu::parse::ParsedFile f2 = slu::parse::parseFile(in2);


			slu::parse::Output<Settings> out2;
			out2.db = in2.genData.mpDb;
			slu::parse::genFile(out2, f2);

			if (out2.text != out.text)
			{
				saveBin((path.parent_path() / "_TEST_OUT"
				            / (path.filename().string() + ".2.d"))
				            .string(),
				    out2.text);

				throw std::runtime_error(
				    "Inconsistant encode of file '" + pathStr + "'");
			}
			log("Encode test success : {}", pathStr);
		} catch (const slu::parse::ParseFailError&)
		//catch (const std::bad_alloc& err)
		{
			logErr("In file {}:", path.string());
			logErr("Encode test failed : {::}", in2.handledErrors);
			return invert ? 2 : 0;
		}
	} catch (const std::runtime_error& e)
	{
		if (!invert || e.what() == "Test failed"sv)
		{
			logErr("In file {}:", path.string());
			if (e.what() != "Test failed"sv)
				logErr("Error msg: {}", e.what());
			logErr("Test failed : {::}", in.handledErrors);
			return 1;
		}
		return 2;
	} catch (const slu::parse::ParseFailError&)
	//catch (const std::bad_alloc& err)
	{
		if (invert)
			return 2;

		logErr("In file {}:", path.string());
		logErr("Test failed : {::}", in.handledErrors);
		return 1;
	}
	return 2;
}

#ifdef _MSC_VER
#ifndef __clang__
extern "C++"
{
#endif
#endif

export int main()
{
#if 0

	constexpr bool TEST_SPEED = false;

	if constexpr (TEST_SPEED)
	{
		const std::vector<uint8_t> srcCode = getBin(
		    "C:/libraries/lua/lua-5.4.4/src/slu/parse/tests/libs/luaunit.lua");

		slu::parse::VecInput in;
		in.fName = "Test";
		in.text = srcCode;

		const auto startTime = std::chrono::system_clock::now();

		for (size_t i = 0; i < 1000; i++)
		{
			in.restart();
			in.genData = {};

			slu::parse::parseFile(in);
		}

		const auto endTime = std::chrono::system_clock::now();

		const auto timeDif
		    = std::chrono::duration_cast<std::chrono::microseconds>(
		        endTime - startTime); // <=104ms, 1014154

		log("Time taken: {}", timeDif);
		log("MB/s: {}",
		    (srcCode.size() * 1000.0) * timeDif.count() / 1000000.0
		        * 0.000001); // div(mb), mul(us 2 s) 	return 0;
	}
#endif

	const std::string_view p = "slu/parse/tests"sv;

	size_t total = 0;
	size_t failed = 0;

	for (auto const& dir_entry :
	    std::filesystem::recursive_directory_iterator{p})
	{
		if (dir_entry.is_directory())
			continue;
		const std::filesystem::path ext = dir_entry.path().extension();
		const bool invert = ext == ".notslu";
		if (ext != ".slu" && !invert)
			continue;

		uint8_t res = testSluOnFile(dir_entry.path(), invert);

		total += (invert || res == 1) ? 1 : 2;
		failed += res == 2 ? 0 : 1;
	}

	log("Tests completed, total (" LUACC_NUM_COL(
	        "{}") ") failed (" LUACC_NUM_COL("{}") ") - " LUACC_NUM_COL("{}") "%",
	    total, failed, 100 - double(100 * failed) / total);


	return (int)failed;
}

#ifdef _MSC_VER
#ifndef __clang__
} // Extern "C++"
#endif
#endif