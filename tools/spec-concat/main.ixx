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
			if (entry.path().string().starts_with("spec/info/"))
				continue;
			if (entry.is_directory())
				continue;
			out += "# Path: " + entry.path().string() + "\n";

			std::ifstream t(entry.path().string());

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
