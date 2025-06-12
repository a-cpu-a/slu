// slu-compiler-api.cpp : Defines the entry point for the application.
//

#include <fstream>
#include <filesystem>
#include <random>
#include <slu/comp/CompInclude.hpp>

#include "slu-compiler-api.h"

std::string canonPath(std::string path)
{
	// Convert to a canonical path, removing any trailing slashes
	try
	{
		path = std::filesystem::weakly_canonical(std::filesystem::path(path)).string();
	}
	catch (const std::filesystem::filesystem_error& e)
	{
		std::cerr << "Error converting path '" << path << "' to canonical form: " << e.what() << "\n";
		return path; // Return the original path if an error occurs
	}
	//Convert to forward slashes
	for (auto& ch : path)
	{
		if (ch == '\\')
			ch = '/';
	}
	// Remove any trailing slashes
	if (!path.empty() && path.back() == '/')
		path.pop_back();

	return path;
}

struct DeletingTmpFile : slu::comp::TmpFile
{
	constexpr DeletingTmpFile(std::string&& path)
		: slu::comp::TmpFile(std::move(path)) {}
	constexpr DeletingTmpFile() = default;
	~DeletingTmpFile() override
	{
		try
		{
			std::filesystem::remove(realPath);
		}
		catch (const std::filesystem::filesystem_error& e)
		{
			std::cerr << "Error deleting temporary file '" << realPath << "': " << e.what() << "\n";
		}
	}
};

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
				files.push_back(canonPath(entry.path().string()));
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
				files.push_back(canonPath(entry.path().string()));
			}
		}
		catch (const std::filesystem::filesystem_error& e)
		{
			std::cerr << "Error reading directory recursively: " << e.what() << "\n";
			return {};
		}
		return files;
		};
	cfg.mkTmpFilePtr = [](std::span<const uint8_t> data) -> slu::comp::TmpFile {
		std::string tmpFileName = "build/_tmp_slu_";

		std::random_device rd;
		std::uniform_int_distribution<short> dist('A', 'Z');
		//Random string
		for (size_t i = 0; i < 28; i++)
		{
			tmpFileName += (char)dist(rd);
		}
		//Create the tmp directory if it doesn't exist
		std::filesystem::create_directories("build/");

		std::ofstream tmpFile(tmpFileName, std::ios::binary);
		if (!tmpFile.is_open())
		{
			std::cerr << "Failed to create temporary file: " << tmpFileName << "\n";
			return {};
		}
		tmpFile.write(reinterpret_cast<const char*>(data.data()), data.size());
		tmpFile.close();
		return DeletingTmpFile(std::move(tmpFileName));
		};

	std::vector<std::string> pathList;
	//All root paths must not end with a slash
	pathList.push_back(canonPath("../hello_world"));
	pathList.push_back(canonPath("../../std"));
	cfg.rootPaths = pathList;
	auto outs = slu::comp::compile(cfg);

	std::filesystem::create_directory("build");

	for (auto& i : outs.entryPoints)
	{
		std::cout << "Entry Point: " << i.fileName << "\n";
		std::cout << "  from: " << i.entryPointFile << "\n";
		// Write the entry point to a file

		std::string outPath = "build/" + i.fileName;
		std::ofstream outFile(outPath, std::ios::binary);
		if (outFile.is_open())
		{
			outFile.write(reinterpret_cast<const char*>(i.contents.data()), i.contents.size());
			outFile.close();
			std::cout << "Written to " << outPath << "\n";
		}
		else
		{
			std::cerr << "Failed to open file for writing: " << i.fileName << "\n";
		}
	}
	std::cout << "Complete.\n";
	return 0;
}
