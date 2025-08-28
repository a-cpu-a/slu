// slu-compiler-api.cpp : Defines the entry point for the application.
//

#ifdef _WIN32
#include <windows.h>
#endif

#include <fstream>
#include <filesystem>
#include <random>
import slu.comp.compile;

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

template<bool msg = true>
struct DeletingTmpFile : slu::comp::TmpFile
{
	constexpr DeletingTmpFile(std::string&& path)
		: slu::comp::TmpFile(std::move(path), destroyFn) {}
	constexpr DeletingTmpFile()
		: slu::comp::TmpFile(destroyFn) {}

	static void destroyFn(TmpFile& thiz)
	{
		if (thiz.realPath.empty())return;
		try
		{
			std::filesystem::remove(thiz.realPath);
		}
		catch (const std::filesystem::filesystem_error& e)
		{
			if (msg)
				std::cerr << "Error deleting temporary file '" << thiz.realPath << "': " << e.what() << "\n";
		}
	}
};

static void(*ctrlCFunc)() = nullptr;
#ifdef _WIN32
static BOOL winCtrlHandler(DWORD)
{
	if (ctrlCFunc != nullptr)
		ctrlCFunc();
	return true;
}
#endif

int main()
{
#ifdef _WIN32

	SetConsoleCtrlHandler(winCtrlHandler, true);

	HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	if (hOut == INVALID_HANDLE_VALUE)
		return 111;
	DWORD dwOriginalOutMode = 0;
	if (!GetConsoleMode(hOut, &dwOriginalOutMode))
		return 104;
	SetConsoleMode(hOut, dwOriginalOutMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);

	SetConsoleOutputCP(CP_UTF8);
#endif

	std::cout << "\x1B[0m";//reset formatting

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
	cfg.errPtr = [](const std::string_view msg) {
		//220 warn
		std::cerr <<"\x1B[38;5;202m" << msg << "\x1B[0m\n";
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
	cfg.mkTmpFilePtr = [](std::optional<std::span<const uint8_t>> data) -> slu::comp::TmpFile {
		std::string tmpFileName = "build/_tmp_slu_";

		std::random_device rd;
		std::uniform_int_distribution<short> dist('A', 'Z');
		//Random string
		for (size_t i = 0; i < 28; i++)
		{
			tmpFileName += (char)dist(rd);
		}
		tmpFileName = canonPath(std::move(tmpFileName));
		//Create the tmp directory if it doesn't exist
		std::filesystem::create_directories("build/");

		if (data.has_value())
		{
			std::ofstream tmpFile(tmpFileName, std::ios::binary);
			if (!tmpFile.is_open())
			{
				std::cerr << "Failed to create temporary file: " << tmpFileName << "\n";
				return {};
			}
			tmpFile.write(reinterpret_cast<const char*>(data->data()), data->size());
			tmpFile.close();
			return DeletingTmpFile(std::move(tmpFileName));
		}

		return DeletingTmpFile<false>(std::move(tmpFileName));
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
