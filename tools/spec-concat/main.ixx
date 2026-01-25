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
	const std::string start_token = "New Syntax {";
	const std::string end_token = "\n}\n";
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

			std::ifstream file(p);

			std::string content((std::istreambuf_iterator<char>(file)),
			    std::istreambuf_iterator<char>());
			std::string_view sv(content);
			while (!sv.empty())
			{
				if (sv.starts_with(start_token))
				{
					//Get rid of that block.
					while (!sv.starts_with(end_token))
						sv = sv.substr(1);
					sv = sv.substr(end_token.size());
					continue;
				}
				out += sv.at(0);
				if (sv.size() == 1)
					break;
				sv = sv.substr(1);
			}
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
