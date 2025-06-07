// slu-compiler-api.cpp : Defines the entry point for the application.
//

#include <fstream>
#include <filesystem>
#include <slu/comp/CompInclude.hpp>

#include "slu-compiler-api.h"
int main()
{
	std::cout << "Hello world!\n";
	

	slu::comp::CompCfg cfg;

	cfg.fileExistsPtr = [](const std::string_view fileName) -> bool {
		return std::filesystem::exists(fileName);
	};
	cfg.getFileContentsPtr = [](const std::string_view fileName) -> std::optional<std::vector<uint8_t>> {
		std::ifstream file(std::string(fileName), std::ios::binary);
		if (!file.is_open())
			return {};

		return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	};
	cfg.isFolderPtr = [](const std::string_view fileName) -> bool {
		return std::filesystem::is_directory(fileName);
	};
	cfg.isSymLinkPtr = [](const std::string_view fileName) -> bool {
		return std::filesystem::is_symlink(fileName);
	};
	cfg.logPtr = [](const std::string_view msg) {
		std::cout << msg << "\n";
	};
	cfg.getFileListPtr = [](const std::string_view folderName) -> std::vector<std::string> {
		std::vector<std::string> files;
		try
		{
			for (const auto& entry : std::filesystem::directory_iterator(folderName))
			{
				files.push_back(entry.path().string());
			}
		}
		catch (const std::filesystem::filesystem_error& e)
		{
			std::cerr << "Error reading directory: " << e.what() << "\n";
			return {};
		}
		return files;
	};
	cfg.getFileListRecPtr = [](const std::string_view folderName) -> std::vector<std::string> {
		std::vector<std::string> files;
		try
		{
			for (const auto& entry : std::filesystem::recursive_directory_iterator(folderName))
			{
				files.push_back(entry.path().string());
			}
		}
		catch (const std::filesystem::filesystem_error& e)
		{
			std::cerr << "Error reading directory recursively: " << e.what() << "\n";
			return {};
		}
		return files;
	};

	std::vector<std::string> pathList;
	//All root paths must not end with a slash
	pathList.push_back("../hello_world");
	pathList.push_back("../../std");
	cfg.rootPaths = pathList;
	auto outs = slu::comp::compile(cfg);

	for (auto& i : outs.entryPoints)
	{
		std::cout << "Entry Point: " << i.fileName << "\n";
		std::cout << "  from: " << i.entryPointFile << "\n";
		// Write the entry point to a file
		std::ofstream outFile(i.fileName, std::ios::binary);
		if (outFile.is_open())
		{
			outFile.write(reinterpret_cast<const char*>(i.contents.data()), i.contents.size());
			outFile.close();
			std::cout << "Written! : " << i.entryPointFile << "\n";
		}
		else
		{
			std::cerr << "Failed to open file for writing: " << i.fileName << "\n";
		}
	}
	std::cout << "Complete.\n";
	return 0;
}
