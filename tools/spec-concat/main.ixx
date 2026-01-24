module;

#include <filesystem>
#include <fstream>
#include <iostream>

export module tools_slu.spec_concat.main;

#ifdef _MSC_VER
#ifndef __clang__
extern "C++"
{
#endif
#endif
export int main()
{
	std::string out;
	try
	{
		for (const auto& entry :
		    std::filesystem::recursive_directory_iterator("spec/"))
		{
			auto p = entry.path().string();
			if (p.starts_with("spec/info/") && !p.ends_with("Usage.txt")
			    && !p.ends_with("Keywords.txt"))
				continue;
			if (entry.is_directory())
				continue;
			out += "# Path: " + p + "\n";

			std::ifstream t(p);

			out += std::string((std::istreambuf_iterator<char>(t)),
			    std::istreambuf_iterator<char>());
			out += '\n';
		}
	} catch (const std::filesystem::filesystem_error& e)
	{
		std::cerr << "Error reading directory recursively: " << e.what()
		          << "\n";
		return 0;
	}
	std::ofstream myfile;
	myfile.open("out/spec-concatenation.txt");
	myfile << out;
	myfile.close();
	return 0;
}
#ifdef _MSC_VER
#ifndef __clang__
extern "C++"
}
#endif
#endif
