// test.cpp : Defines the entry point for the application.
//

#include <format>
#include <filesystem>
#include <span>
#include <fstream>
#include <iostream>
#include <slu/parser/Parse.hpp>
#include <slu/paint/Paint.hpp>
#include <slu/paint/SemOutputStream.hpp>
#include <slu/paint/StyledColorConverter.hpp>
#include <slu/paint/PaintToHtml.hpp>
#include <slu/gen/Gen.hpp>

template<class... Args>
inline void log(const std::format_string<Args...> fmt, Args&&... fmtArgs)
{
	std::cout << LUACC_DEFAULT << std::vformat(fmt.get(), std::make_format_args(fmtArgs...))
		<< "\n";
}
template<class... Args>
inline void logErr(const std::format_string<Args...> fmt, Args&&... fmtArgs)
{
	std::cerr << LUACC_INVALID << std::vformat(fmt.get(), std::make_format_args(fmtArgs...))
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
	std::ofstream out(path,std::ios::binary);
	if (!out)
	{
		logErr("Failed to open file for writing: {}", path);
		return;
	}
	out.write(reinterpret_cast<const char*>(data.data()), data.size());
}

template<bool slu>
inline uint8_t testSluOnFile(const std::filesystem::path& path, const bool invert)
{
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

	using Settings = slu::parse::Sel<slu, slu::parse::Setting<void>, decltype(slu::parse::sluCommon)>;

	slu::parse::BasicMpDbData dbData;
	slu::parse::VecInput<Settings> in;
	in.fName = pathStr;
	in.text = srcCode;
	if constexpr (slu)
		in.genData.mpDb.data = &dbData;

	try
	{
		//_ASSERT(!path.filename().string().ends_with("concatenated.slu"));

		slu::parse::ParsedFile<decltype(in)> f = slu::parse::parseFile(in);


		if (invert)
		{
			throw std::runtime_error("Test failed");
		}
		else
			log("Test success : {}", pathStr);

		slu::paint::SemOutput<decltype(in), slu::paint::StyledColorConverter<false>> se = { in };
		in.restart();
		slu::paint::paintFile(se, f);
		in.restart();
		const std::string outHtml = slu::paint::toHtml(se, true);

		slu::parse::Output<Settings> out;
		out.db = std::move(in.genData.mpDb);
		slu::parse::genFile(out, f);


		saveBin((path.parent_path() / "_TEST_OUT" / (path.filename().string() + ".d")).string(), out.text);

		slu::parse::VecInput<Settings> in2;
		if constexpr (slu)
			in2.genData.mpDb.data = &dbData;
		in2.fName = pathStr;
		in2.text = out.text;
		in2.restart();

		try
		{

			slu::parse::ParsedFile<decltype(in2)> f2 = slu::parse::parseFile(in2);


			slu::parse::Output<Settings> out2;
			out2.db = in2.genData.mpDb;
			slu::parse::genFile(out2, f2);

			if (out2.text != out.text)
			{
				saveBin((path.parent_path() / "_TEST_OUT" / (path.filename().string() + ".2.d")).string(), out2.text);

				throw std::runtime_error("Inconsistant encode of file '" + pathStr + "'");
			}
			log("Encode test success : {}", pathStr);
		}
		catch (const slu::parse::ParseFailError&)
			//catch (const std::bad_alloc& err)
		{
			logErr("In file {}:", path.string());
			logErr("Encode test failed : {::}", in2.handledErrors);
			return invert ? 2 : 0;
		}
	}
	catch (const std::runtime_error& e)
	{
		if (!invert || e.what() == "Test failed")
		{
			logErr("In file {}:", path.string());
			logErr("Test failed : {::}", in.handledErrors);
			return 1;
		}
		return 2;
	}
	catch (const slu::parse::ParseFailError&)
		//catch (const std::bad_alloc& err)
	{
		if (invert)return 2;

		logErr("In file {}:", path.string());
		logErr("Test failed : {::}", in.handledErrors);
		return 1;
	}
	return 2;
}


int main()
{
	//const auto x =slu::spec::extract_and_merge_ebnf_blocks("C:/libraries/lua/lua-5.4.4/src/slua/spec/");


	const bool TEST_SPEED = false;

	if (TEST_SPEED)
	{
		const std::vector<uint8_t> srcCode = getBin("C:/libraries/lua/lua-5.4.4/src/slu/parser/tests/libs/luaunit.lua");

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

		const auto timeDif = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);// <=104ms, 1014154

		log("Time taken: {}", timeDif);
		log("MB/s: {}", (srcCode.size() * 1000.0) * timeDif.count() / 1000000.0 * 0.000001);// div(mb), mul(us 2 s)
		return 0;
	}

	std::u8string src = u8"return ... + 4456.546456 & not c89'igma\\z  333'  \n\n\r\r";

	slu::parse::VecInput in;
	in.fName = "Test.lua";
	in.text = std::bit_cast<std::span<const uint8_t>>(std::span(src));

	slu::parse::ParsedFile<decltype(in)> f = slu::parse::parseFile(in);

	const std::string_view p2 = "C:/libraries/lua/lua-5.4.4/src/slu/ext/lua/testes";
	const std::string_view p = "C:/libraries/lua/lua-5.4.4/src/slu/parser/tests";

	size_t total = 0;
	size_t failed = 0;

	for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ p })
	{
		if (dir_entry.is_directory())
			continue;
		const std::filesystem::path ext = dir_entry.path().extension();
		const bool invert = ext == ".notlua" || ext == ".notslu";
		if (ext != ".lua" && ext != ".slu" && !invert)
			continue;
		const bool slu = ext.string().ends_with("slu");

		uint8_t res;
		if (slu)
			res = testSluOnFile<true>(dir_entry.path(), invert);
		else
			res = testSluOnFile<false>(dir_entry.path(), invert);

		total += (invert || res == 1) ? 1 : 2;
		failed += res == 2 ? 0 : 1;
	}
	for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ p2 })
	{
		if (dir_entry.is_directory())
			continue;
		const std::filesystem::path ext = dir_entry.path().extension();
		const bool invert = ext == ".notlua" || ext == ".notslu";
		if (ext != ".lua" && ext != ".slu" && !invert)
			continue;
		const bool slu = ext.string().ends_with("slu");

		uint8_t res;
		if (slu)
			res = testSluOnFile<true>(dir_entry.path(), invert);
		else
			res = testSluOnFile<false>(dir_entry.path(), invert);

		total += (invert || res == 1) ? 1 : 2;
		failed += res == 2 ? 0 : 1;
	}

	log(
		"Test completed, total("
		LUACC_NUM_COL("{}")
		") failed("
		LUACC_NUM_COL("{}")
		") - "
		LUACC_NUM_COL("{}") "%",
		total, failed, 100 - double(100 * failed) / total);


	return (int)failed;
}
